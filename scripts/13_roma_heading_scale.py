"""Faz 1B sağlaması (sayısal): RoMa v2 ile yön (Phi) tanımı ve ölçek s.

Her seçilen kare için drone görüntüsü 0/90/180/270° döndürülerek kuzey-yukarı uydu
karosuyla eşleştirilir; en çok inlier'lı yönde drone → karo benzerlik dönüşümü
(cv2.estimateAffinePartial2D, RANSAC) kestirilir. Buradan:
  * bearing: drone görüntüsünün ÜST kenarının kuzeye göre açısı (saat yönü, derece)
  * m_per_px: tam çözünürlüklü drone pikseli başına metre  (→ ölçek s ve ayak izi)
  * center_off_m: drone görüntü merkezinin karo merkezine (CSV lat/lon) uzaklığı
Phi1/Phi2 ile bearing arasındaki fark bölge bazında özetlenir (MIMARI.md §1B, §13 B planı:
"yön sütunu tanımı belirsiz → RoMa v2 homografisinden açı çıkar, sütunla karşılaştır").

Kullanım:
  python scripts/13_roma_heading_scale.py data/UAV_VisLoc_dataset --per-region 12 --tile-m 1000
Çıktı: results/faz1/roma_heading_scale.csv, results/faz1/roma_heading_scale_summary.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc.geo import angular_error_deg
from guvenloc.io_visloc import crop_square_m, list_regions, load_region
from guvenloc.metrics import log_run

DRONE_DOWNSCALE = 4  # 3976x2652 → 994x663; RoMa zaten 640'a küçültüyor


def rot90_img(img: np.ndarray, k: int) -> np.ndarray:
    """np.rot90: saat yönünün tersine k×90°."""
    return np.ascontiguousarray(np.rot90(img, k))


def unrot90_pts(pts: np.ndarray, k: int, w: int, h: int) -> np.ndarray:
    """rot90_img(img, k) üzerindeki noktaları orijinal görüntü (w×h) koordinatına geri çevirir."""
    x, y = pts[:, 0], pts[:, 1]
    k %= 4
    if k == 0:
        return pts.copy()
    if k == 1:   # (x, y) → (y, w-1-x)
        return np.stack([w - 1 - y, x], 1)
    if k == 2:   # (x, y) → (w-1-x, h-1-y)
        return np.stack([w - 1 - x, h - 1 - y], 1)
    return np.stack([y, h - 1 - x], 1)  # k == 3: (x, y) → (h-1-y, x)


def similarity_params(M: np.ndarray) -> tuple[float, float]:
    """[[a,-b,tx],[b,a,ty]] → (ölçek, bearing°). Görüntü y aşağı olduğundan
    drone 'yukarı' vektörü (0,-1) karoda (b,-a)'ya gider; kuzeye göre saat yönü açı atan2(b, a)."""
    a, b = M[0, 0], M[1, 0]
    return float(np.hypot(a, b)), float(np.degrees(np.arctan2(b, a)) % 360.0)


def circ_mean_deg(x) -> float:
    r = np.radians(np.asarray(x, float))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360.0)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--regions", nargs="*")
    ap.add_argument("--per-region", type=int, default=12)
    ap.add_argument("--tile-m", type=float, default=1000.0)
    ap.add_argument("--tile-px", type=int, default=1024)
    ap.add_argument("--setting", default="base")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--ransac-px", type=float, default=4.0, help="karo pikseli")
    ap.add_argument("--min-inliers", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/faz1"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from romav2 import RoMaV2

    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(args.seed)
    model = RoMaV2()
    model.apply_setting(args.setting)

    rows = []
    t0 = time.perf_counter()
    for rid in args.regions or list_regions(args.root):
        reg = load_region(args.root, rid)
        n = len(reg.frames)
        # ilk/son %5 atlanır (kalkış, su üstü); kalan aralık eşit örneklenir
        idx = np.unique(np.linspace(0.05 * n, 0.95 * (n - 1), args.per_region).round().astype(int))
        for i in idx:
            row = reg.frames.iloc[int(i)]
            img = cv2.imread(str(reg.frame_path(str(row["filename"]))))
            if img is None:
                continue
            H0, W0 = img.shape[:2]
            img = cv2.resize(img, (W0 // DRONE_DOWNSCALE, H0 // DRONE_DOWNSCALE),
                             interpolation=cv2.INTER_AREA)
            drone = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = drone.shape[:2]
            tile = np.ascontiguousarray(
                crop_square_m(reg, float(row["lat"]), float(row["lon"]), args.tile_m, args.tile_px)
            )
            best = None
            for k in range(4):
                dk = rot90_img(drone, k)
                preds = model.match(dk, tile)
                matches, *_ = model.sample(preds, args.n)
                kA, kB = (t.detach().float().cpu().numpy() for t in model.to_pixel_coordinates(
                    matches, dk.shape[0], dk.shape[1], tile.shape[0], tile.shape[1]))
                kA = unrot90_pts(kA, k, w, h)
                M, mask = cv2.estimateAffinePartial2D(
                    kA, kB, method=cv2.RANSAC, ransacReprojThreshold=args.ransac_px,
                    maxIters=5000, confidence=0.999)
                inl = int(mask.sum()) if mask is not None else 0
                if M is not None and (best is None or inl > best[0]):
                    best = (inl, k, M)
            rec = {"region": rid, "num": int(row["num"]), "filename": row["filename"],
                   "height": float(row["height"]),
                   "phi1": float(row["Phi1"]) if "Phi1" in row else np.nan,
                   "phi2": float(row["Phi2"]) if "Phi2" in row else np.nan,
                   "inliers": 0, "rot_k": -1}
            if best is not None:
                inl, k, M = best
                scale, bearing = similarity_params(M)       # karo px / küçültülmüş drone px
                m_per_tile_px = args.tile_m / args.tile_px
                m_per_px = scale * m_per_tile_px / DRONE_DOWNSCALE
                c = M @ np.array([w / 2, h / 2, 1.0])
                off = (c - args.tile_px / 2) * m_per_tile_px
                rec.update(inliers=inl, rot_k=k, bearing=bearing, m_per_px=m_per_px,
                           footprint_w_m=m_per_px * W0, footprint_h_m=m_per_px * H0,
                           center_off_e_m=off[0], center_off_s_m=off[1],
                           center_off_m=float(np.hypot(*off)))
                if np.isfinite(rec["phi1"]):
                    rec["d_phi1"] = (bearing - rec["phi1"]) % 360.0
                    rec["d_phi2"] = (bearing - rec["phi2"]) % 360.0
            rows.append(rec)
            print(f"{rid} #{rec['num']:4d} inl={rec['inliers']:5d} "
                  f"bearing={rec.get('bearing', np.nan):6.1f} phi1={rec['phi1']:7.1f} "
                  f"d={rec.get('d_phi1', np.nan):6.1f} m/px={rec.get('m_per_px', np.nan):.3f} "
                  f"off={rec.get('center_off_m', np.nan):6.1f} m", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "roma_heading_scale.csv", index=False)

    good = df[df.inliers >= args.min_inliers]
    summ = []
    for rid, g in good.groupby("region"):
        s = {"region": rid, "n_ok": len(g), "n": int((df.region == rid).sum()),
             "height_med": g.height.median(), "m_per_px_med": g.m_per_px.median(),
             "footprint_w_med_m": g.footprint_w_m.median(),
             "k_footprint_over_height": (g.footprint_w_m / g.height).median(),
             "center_off_med_m": g.center_off_m.median()}
        if g.d_phi1.notna().any() if "d_phi1" in g else False:
            mu = circ_mean_deg(g.d_phi1.dropna())
            s["d_phi1_circmean"] = mu
            s["d_phi1_spread"] = float(np.median([angular_error_deg(d, mu) for d in g.d_phi1.dropna()]))
            mu2 = circ_mean_deg(g.d_phi2.dropna())
            s["d_phi2_circmean"] = mu2
            s["d_phi2_spread"] = float(np.median([angular_error_deg(d, mu2) for d in g.d_phi2.dropna()]))
        summ.append(s)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(args.out / "roma_heading_scale_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(sdf.round(2).to_string(index=False))
    print(f"süre: {time.perf_counter() - t0:.0f} s")

    cfg = Path("configs/base.yaml")
    for _, s in sdf.iterrows():
        if "d_phi1_spread" in s and np.isfinite(s.get("d_phi1_spread", np.nan)):
            log_run("13_roma_heading_scale.py", cfg, args.seed, f"region{s.region}",
                    "d_phi1_spread_deg", round(float(s.d_phi1_spread), 2),
                    notes=f"tile_m={args.tile_m} setting={args.setting}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
