"""Faz 0.4 – RoMa v2 hassasiyet (precision) matrisinin birimi.

RoMa v2 hassasiyeti P'yi Cholesky parametreleriyle "piksel" biriminde kurar
(refiner.py: `l00 = softplus(.) + eps  # this is in pixels`) ve
`RoMaV2.prec_map_coordinates(P, H_in, W_in, H_out, W_out)` ile çözünürlükler arasında
P·(in/out)² ölçekler. Hangi çözünürlüğün pikseli olduğu deneyle belirlenir:

  Gerçek uydu karoları A, bilinen homografi H ile B = H(A) (+ fotometrik bozulma).
  Yoğun warp_AB ızgarasının her noktasında e = tahmin(B) − H(A) (B pikseli) bilinir.
  Doğru birimde P_B için m = eᵀ P_B e ≈ χ²(2): ortalama 2, medyan 1,386, %95 < 5,99.

Hipotezler (R = P'nin tanımlı olduğu ızgara): lr, hr, anchor (512), imgB (B görüntüsü),
norm (normalize [-1,1] koordinatı; R = 2).
Kullanım:
  python scripts/02_precision_units.py data/UAV_VisLoc_dataset --setting base
  python scripts/02_precision_units.py data/UAV_VisLoc_dataset --setting mega1500
Çıktı: results/faz0/precision_units_<setting>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc.io_visloc import crop_square_m, load_region
from guvenloc.metrics import log_run

CHI2_2_MEDIAN = 2 * np.log(2)  # 1,386


def random_homography(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """Merkez etrafında dönme ±30°, ölçek 0,8–1,25, hafif perspektif, öteleme."""
    ang = np.radians(rng.uniform(-30, 30))
    s = np.exp(rng.uniform(np.log(0.8), np.log(1.25)))
    c = np.array([w / 2, h / 2])
    R = s * np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    H = np.eye(3)
    H[:2, :2] = R
    H[:2, 2] = c - R @ c + rng.uniform(-0.05, 0.05, 2) * [w, h]
    H[2, :2] = rng.uniform(-1.5e-4, 1.5e-4, 2)
    return H


def photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = img.astype(np.float32)
    x = x * rng.uniform(0.75, 1.25, 3) + rng.uniform(-20, 20, 3)
    x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 1.2))
    x += rng.normal(0, 4, x.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--setting", default="base")
    ap.add_argument("--regions", nargs="*", default=["01", "04", "08", "09"])  # yalnızca train
    ap.add_argument("--pairs-per-region", type=int, default=5)
    ap.add_argument("--min-overlap", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from romav2 import RoMaV2
    from romav2.geometry import get_normalized_grid, to_pixel

    torch.set_float32_matmul_precision("highest")
    model = RoMaV2()
    model.apply_setting(args.setting)
    rng = np.random.default_rng(args.seed)
    WA, HA, WB, HB = 900, 600, 700, 700  # farklı boyutlar: hipotezler ayrışsın

    m_all = {k: [] for k in ("lr", "hr", "anchor", "imgB", "norm")}
    err_all, sig_raw_all = [], []
    for rid in args.regions:
        reg = load_region(args.root, rid)
        for _ in range(args.pairs_per_region):
            row = reg.frames.iloc[int(rng.integers(len(reg.frames)))]
            big = np.ascontiguousarray(crop_square_m(reg, row.lat, row.lon, 900, 1200))
            A = np.ascontiguousarray(big[300:300 + HA, 150:150 + WA])
            H = random_homography(WA, HA, rng)
            # B = H(A), B boyutu WB×HB; B'nin merkezini A'nın merkezine hizala
            T = np.array([[1, 0, (WB - WA) / 2], [0, 1, (HB - HA) / 2], [0, 0, 1]])
            Hf = T @ H
            B = photometric(cv2.warpPerspective(A, Hf, (WB, HB), flags=cv2.INTER_LINEAR), rng)

            preds = model.match(A, B)
            warp = preds["warp_AB"][0]                       # (Hg, Wg, 2) normalize
            P = preds["precision_AB"][0].double()            # (Hg, Wg, 2, 2) ham
            ov = preds["overlap_AB"][0, ..., 0]
            Hg, Wg = warp.shape[:2]
            grid = get_normalized_grid(1, Hg, Wg, overload_device=warp.device)[0]
            xa = to_pixel(grid, H=HA, W=WA).cpu().numpy().reshape(-1, 2) - 0.5  # cv2 merkezleri
            xb = to_pixel(warp, H=HB, W=WB).cpu().numpy().reshape(-1, 2) - 0.5
            gt = cv2.perspectiveTransform(xa[None].astype(np.float64), Hf)[0]
            inside = (gt[:, 0] > 2) & (gt[:, 0] < WB - 3) & (gt[:, 1] > 2) & (gt[:, 1] < HB - 3)
            keep = inside & (ov.cpu().numpy().reshape(-1) > args.min_overlap)
            e = (xb - gt)[keep]
            Pr = P.cpu().numpy().reshape(-1, 2, 2)[keep]
            sel = rng.choice(len(e), min(4000, len(e)), replace=False)
            e, Pr = e[sel], Pr[sel]
            res = {"lr": (model.W_lr, model.H_lr),
                   "hr": (model.W_hr or model.W_lr, model.H_hr or model.H_lr),
                   "anchor": (model.anchor_width, model.anchor_height),
                   "imgB": (WB, HB), "norm": (2, 2)}
            for k, (wr, hr) in res.items():
                D = np.diag([wr / WB, hr / HB])               # P_B = D P_R D
                PB = D @ Pr @ D
                m_all[k].append(np.einsum("ni,nij,nj->n", e, PB, e))
            err_all.append(np.linalg.norm(e, axis=1))
            sig_raw_all.append(np.sqrt(np.trace(np.linalg.inv(Pr), axis1=1, axis2=2) / 2))
            print(f"{rid}: grid {Wg}x{Hg}, {keep.sum()} geçerli, medyan |e| "
                  f"{np.median(err_all[-1]):.2f} px", flush=True)

    err = np.concatenate(err_all)
    sig_raw = np.concatenate(sig_raw_all)
    out = {"setting": args.setting, "grid_lr": [model.W_lr, model.H_lr],
           "grid_hr": [model.W_hr, model.H_hr], "img_B": [WB, HB],
           "n": len(err), "err_px_median": float(np.median(err)),
           "err_px_p90": float(np.percentile(err, 90)), "hypotheses": {}}
    print(f"\n{args.setting}: n={len(err)}, medyan |e| = {np.median(err):.3f} px (B görüntüsü)")
    print(f"{'hipotez':8s} {'medyan m':>9s} {'ort. m (≤%99)':>14s} {'P(m<5,99)':>10s} "
          f"{'medyan σ_B px':>14s}   (χ²(2): 1,386 / 2,0 / 0,95)")
    for k, ms in m_all.items():
        m = np.concatenate(ms)
        wr = {"lr": model.W_lr, "hr": model.W_hr or model.W_lr, "anchor": model.anchor_width,
              "imgB": WB, "norm": 2}[k]
        sig_b = sig_raw * WB / wr
        trimmed = m[m <= np.percentile(m, 99)]
        h = {"median_m": float(np.median(m)), "mean_m_trim99": float(trimmed.mean()),
             "frac_below_5.99": float((m < 5.99).mean()),
             "sigma_B_px_median": float(np.median(sig_b)),
             "log_ratio_to_chi2_median": float(np.log(np.median(m) / CHI2_2_MEDIAN))}
        out["hypotheses"][k] = h
        print(f"{k:8s} {h['median_m']:9.3f} {h['mean_m_trim99']:14.3f} "
              f"{h['frac_below_5.99']:10.3f} {h['sigma_B_px_median']:14.3f}")
    # Ayrıca: |e| ile σ korelasyonu (belirsizlik bilgilendirici mi?)
    rho = float(np.corrcoef(np.log(sig_raw + 1e-12), np.log(err + 1e-6))[0, 1])
    out["corr_log_sigma_log_err"] = rho
    print(f"corr(log σ, log |e|) = {rho:.3f}")

    best = min(out["hypotheses"], key=lambda k: abs(out["hypotheses"][k]["log_ratio_to_chi2_median"]))
    out["best"] = best
    print(f"χ²(2)'ye en yakın hipotez: {best}")
    Path("results/faz0").mkdir(parents=True, exist_ok=True)
    p = Path(f"results/faz0/precision_units_{args.setting}.json")
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log_run("02_precision_units.py", Path("configs/base.yaml"), args.seed, "train",
            f"precision_median_m_{best}", round(out["hypotheses"][best]["median_m"], 4),
            notes=f"setting={args.setting}")
    print("rapor:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
