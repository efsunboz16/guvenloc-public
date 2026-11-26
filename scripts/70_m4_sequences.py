"""Faz 5A – uçuş dizileri: kare arası seyreklik ve drone↔drone hareket kestirimi (tahmin adımının girdisi).

  python scripts/70_m4_sequences.py          # eğitim + calib bölgeleri (test 02, 03, 06: Faz 6'ya kadar kapalı)

Her bölgede pozitif kareler dosya sırasıyla dizilir (önbellek seq/<bölge>_<t>_<t+1>.npz ile aynı sıra).
Ardışık kare çifti başına:
  · GPS: yer mesafesi d (m, CSV lat/lon), süre Δt (s), yön farkı ΔΦ1 (°), ayak izi örtüşmesi ≈ 1 − d / L
    (L = karenin uçuş yönündeki kenarı ≈ yükseklik · drone_m_per_px).
  · Görüntü: önbellekteki RoMa v2 eşleşmelerinden homografi (baselines.fit_homography, 5 px). t+1 merkezinin t'deki
    karşılığı c' = H⁻¹(c) → drone çerçevesinde (ileri, sağ) piksel → × drone_m_per_px → kuzey/doğu (Φ1_t ile).
    Δθ = homografinin yön farkı. Karşılaştırma: GPS yer değiştirmesi ve ΔΦ1.
Üçgen denetimi (F5-2): iki ardışık karenin de uyduyla iyi eşleştiği yerlerde (en çok inlier'lı dönme, inlier ≥ 500)
uydu eşleşmesinin (RANSAC merkezi → enlem/boylam) yer değiştirmesi; görüntü↔uydu, görüntü↔GPS, uydu↔GPS farkları.
→ results/faz5/m4/seq_stats.csv, seq_summary.md / .json
"""
from __future__ import annotations

import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc import baselines as B
from guvenloc.cache import seq_file
from guvenloc.geo import EARTH_R
from guvenloc.matcher import MatchSet

OUT = Path("results/faz5/m4")
TEST = {"02", "03", "06"}
DATA = Path("data/UAV_VisLoc_dataset")


def enu(lat0, lon0, lat, lon):
    """(doğu, kuzey) m, lat0/lon0'a göre (eşdikdörtgen; birkaç km için yeterli)."""
    return (math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0)),
            math.radians(lat - lat0) * EARTH_R)


def img_to_en(fwd: float, right: float, phi_deg: float) -> tuple[float, float]:
    """Drone çerçevesi (ileri = görüntü üstü, sağ) → (doğu, kuzey); φ = görüntü üst kenarının pusula yönü."""
    p = math.radians(phi_deg)
    return fwd * math.sin(p) + right * math.cos(p), fwd * math.cos(p) - right * math.sin(p)


def seq_motion(path: Path, ransac_px: float = 5.0) -> dict:
    """t → t+1 homografisinden drone çerçevesinde (ileri, sağ) piksel ve Δθ (°, saat yönü)."""
    out = {"n_matches": 0, "n_inliers": 0, "fwd_px": np.nan, "right_px": np.nan, "dtheta": np.nan}
    if not path.exists():
        return out
    ms = MatchSet.load_npz(path)
    fit = B.fit_homography(ms, ransac_px=ransac_px)
    out.update(n_matches=fit["n_matches"], n_inliers=fit["n_inliers"])
    if fit["H"] is None:
        return out
    w, h = ms.meta["drone_wh"]
    c = np.array([[(w - 1) / 2, (h - 1) / 2]])
    try:
        Hi = np.linalg.inv(fit["H"])
    except np.linalg.LinAlgError:
        return out
    cp = B.project(Hi, c)[0]                 # t+1 merkezi, t karesinde
    d = cp - c[0]
    # yön: t+1'in "yukarı"sı t'de hangi yöne bakıyor (t'nin üstüne göre, saat yönü)
    up = B.project(Hi, c + [[0, -0.05 * h]])[0] - cp
    out.update(fwd_px=float(-d[1]), right_px=float(d[0]),
               dtheta=float(math.degrees(math.atan2(up[0], -up[1]))))
    return out


def triangle(d: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    from guvenloc.pipeline import tile_px_to_latlon
    p = pairs.set_index("pair_id")
    f = pd.concat([pd.read_csv(f"results/faz1/baselines/feats_v1_{s}_px5.csv") for s in ("train", "calib")],
                  ignore_index=True)
    f = f[f.pair_id.str.endswith("_pos")]
    b = f.loc[f.groupby("pair_id").n_inliers.idxmax()].set_index("pair_id").join(p)
    b = b[b.n_inliers >= 500]
    b["slat"], b["slon"] = zip(*[tile_px_to_latlon(r.cx, r.cy, r.tile_lat, r.tile_lon, r.tile_size_m, int(r.tile_px))
                                 for r in b.itertuples()], strict=True)
    d = d.assign(pa=d.region + "_" + d.t0.map("{:04d}".format) + "_pos",
                 pb=d.region + "_" + d.t1.map("{:04d}".format) + "_pos")
    d = d[d.pa.isin(b.index) & d.pb.isin(b.index) & (d.n_inliers >= 50)].copy()
    A, Bb = b.loc[d.pa], b.loc[d.pb]
    de = np.radians(Bb.slon.to_numpy() - A.slon.to_numpy()) * EARTH_R * np.cos(np.radians(A.slat.to_numpy()))
    dn = np.radians(Bb.slat.to_numpy() - A.slat.to_numpy()) * EARTH_R
    return pd.DataFrame({"region": d.region.to_numpy(),
                         "img_sat": np.hypot(d.de_img - de, d.dn_img - dn), "img_gps": d.motion_err_m.to_numpy(),
                         "sat_gps": np.hypot(de - d.de_gps, dn - d.dn_gps)})


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pairs = pd.read_csv("data/pairs/pairs_v1.csv", dtype={"region": str, "tile_region": str})
    pos = pairs[(pairs.label == "pos") & ~pairs.region.isin(TEST)].sort_values(["region", "drone_path"])
    rows = []
    for rid, g in pos.groupby("region"):
        csv = pd.read_csv(next((DATA / rid).glob("*.csv")))
        csv["t"] = pd.to_datetime(csv.date)
        meta = csv.set_index("filename")
        g = g.assign(num=g.pair_id.str.split("_").str[1].astype(int),
                     fn=g.drone_path.str.split("/").str[-1]).reset_index(drop=True)
        jobs = list(zip(g.iloc[:-1].itertuples(), g.iloc[1:].itertuples(), strict=True))
        files = [seq_file("cache/v1", rid, a.num, b.num) for a, b in jobs]
        with ThreadPoolExecutor(8) as ex:
            mots = list(ex.map(seq_motion, files))
        for (a, b), m in zip(jobs, mots, strict=True):
            ma, mb = meta.loc[a.fn], meta.loc[b.fn]
            de, dn = enu(a.drone_lat, a.drone_lon, b.drone_lat, b.drone_lon)
            mpp = float(a.drone_m_per_px)
            L = 2652 * mpp                                     # uçuş yönündeki kenar (kısa kenar, üst = ileri)
            e_img, n_img = (img_to_en(m["fwd_px"] * mpp, m["right_px"] * mpp, ma.Phi1)
                            if np.isfinite(m["fwd_px"]) else (np.nan, np.nan))
            dphi = (mb.Phi1 - ma.Phi1 + 180) % 360 - 180
            rows.append({"region": rid, "split": a.split, "t0": a.num, "t1": b.num, "gap": b.num - a.num,
                         "dist_m": math.hypot(de, dn), "dt_s": (mb.t - ma.t).total_seconds(),
                         "overlap": max(0.0, 1 - math.hypot(de, dn) / L), "dphi_gps": dphi,
                         **m, "de_img": e_img, "dn_img": n_img, "de_gps": de, "dn_gps": dn,
                         "motion_err_m": math.hypot(e_img - de, n_img - dn) if np.isfinite(e_img) else np.nan,
                         "dtheta_err": abs((m["dtheta"] - dphi + 180) % 360 - 180) if np.isfinite(m["dtheta"])
                         else np.nan})
        print(f"{rid}: {len(jobs)} ardışık çift", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "seq_stats.csv", index=False)
    ok = df.n_inliers >= 50
    summ = df.assign(ok=ok).groupby("region").agg(
        split=("split", "first"), n=("t0", "size"), gap_gt1=("gap", lambda x: int((x > 1).sum())),
        dist_med=("dist_m", "median"), dt_med=("dt_s", "median"), overlap_med=("overlap", "median"),
        no_overlap=("overlap", lambda x: float((x <= 0).mean())), inl_med=("n_inliers", "median"),
        ok=("ok", "mean"), merr_med=("motion_err_m", "median"),
        merr_p90=("motion_err_m", lambda x: float(x.quantile(0.9))),
        dth_med=("dtheta_err", "median"))
    L = ["# Faz 5A — uçuş dizileri (eğitim + calib; test kapalı)", "",
         ("Ardışık pozitif kareler. Hareket hatası = |homografi yer değiştirmesi − GPS yer değiştirmesi| (m), "
         "Δθ hatası = |homografi yön farkı − ΔΦ1| (°). 'kullanılır' = homografi inlier ≥ 50."), "",
         ("| bölge | küme | çift | atlama > 1 | mesafe medyan | Δt medyan | örtüşme medyan | örtüşmesiz | inlier medyan | "
         "kullanılır | hareket hatası medyan / %90 | Δθ hatası medyan |"), "|---|" * 1 + "---|" * 11]
    for r, s in summ.iterrows():
        L.append(f"| {r} | {s.split} | {s.n} | {s.gap_gt1} | {s.dist_med:.0f} m | {s.dt_med:.0f} s | {s.overlap_med:.2f} | "
                 f"{s.no_overlap:.3f} | {s.inl_med:.0f} | {s.ok:.3f} | {s.merr_med:.1f} / {s.merr_p90:.1f} m | "
                 f"{s.dth_med:.2f}° |")
    u = df[ok]
    L += ["", (f"Toplam {len(df)} çift; kullanılır {ok.mean():.3f}. Kullanılır çiftlerde hareket hatası medyan "
          f"{u.motion_err_m.median():.1f} m (%90 {u.motion_err_m.quantile(0.9):.1f} m), mesafeye oranı medyan "
          f"{(u.motion_err_m / u.dist_m).median():.3f}; Δθ hatası medyan {u.dtheta_err.median():.2f}°.")]
    tri = triangle(df, pairs)
    tm = tri.groupby("region")[["img_sat", "img_gps", "sat_gps"]].median()
    L += ["", f"## Üçgen denetimi (F5-2): {len(tri)} ardışık çift, iki kare de uyduyla iyi eşleşmiş", "",
          "| bölge | görüntü ↔ uydu | görüntü ↔ GPS | uydu ↔ GPS |", "|---|---|---|---|"]
    L += [f"| {r} | {s.img_sat:.1f} m | {s.img_gps:.1f} m | {s.sat_gps:.1f} m |" for r, s in tm.iterrows()]
    L.append(f"| hepsi | {tri.img_sat.median():.1f} m | {tri.img_gps.median():.1f} m | {tri.sat_gps.median():.1f} m |")
    (OUT / "seq_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "seq_summary.json").write_text(summ.to_json(orient="index", force_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
