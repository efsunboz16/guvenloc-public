"""Faz 1 K1 – GT kaymasının kamera eğimiyle açıklanması (önceden konmuş kural, faz1_durum.md K1).

  python scripts/41_gt_tilt.py --cache cache/v1

Kural (değiştirilmeden uygulanır):
  * Eğitim bölgelerinde, satır 2'nin (4 yön + en çok inlier) KABUL ettiği ve hatası < 100 m olan
    pozitiflerde RoMa'nın bulduğu drone görüntü merkezi ile CSV lat/lon arasındaki kayma,
    drone çerçevesine (ileri/yanal, Phi1 ile) çevrilir.
  * Doğrusal model `yanal ~ a + b·Omega + c·Kappa`, `ileri ~ a + b·Omega + c·Kappa`
    YALNIZCA eğitimde kurulur (en küçük kareler).
  * calib'de (aynı seçim kuralı) medyan kayma büyüklüğü öncesi/sonrası. ≥ %30 azalırsa
    düzeltilmiş GT ikincil sonuç olarak `pairs_v2`; azalmazsa konu kapanır.
Ek bilgi olarak (karar için KULLANILMAZ): AGL'ye bölünmüş (açısal) model ve bölge kırılımı.

Eksenler: karo kuzey-yukarı (x doğu, y güney). Drone çerçevesi: ileri = görüntü üst kenarının
yönü (pusula açısı Phi1), yanal = sağ. Kayma = RoMa merkezi − CSV konumu (metre).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc.io_visloc import _read_csv, find_root
from guvenloc.metrics import log_run

OUT = Path("results/faz1")
MAX_ERR_M = 100.0
RULE_REDUCTION = 0.30
BLOCK = 50


def selected_positives(feats: pd.DataFrame, pairs: pd.DataFrame, tau2: int) -> pd.DataFrame:
    """Satır 2 seçimi (en çok inlier), kabul (≥ τ2) ve hata < 100 m olan pozitifler."""
    pos = pairs[pairs.label == "pos"].set_index("pair_id")
    f = feats[feats.pair_id.isin(pos.index)]
    f = f.groupby("pair_id").filter(lambda g: set(g.rot) >= {0, 90, 180, 270})
    best = f.loc[f.groupby("pair_id").n_inliers.idxmax()].set_index("pair_id")
    best = best[best.valid.astype(bool) & (best.n_inliers >= tau2)]
    d = best.join(pos, how="inner")
    mpp = d.tile_size_m / d.tile_px
    d["de_m"] = (d.cx - d.gt_x) * mpp            # doğu
    d["dn_m"] = -(d.cy - d.gt_y) * mpp           # kuzey (y aşağı = güney)
    d["err_m"] = np.hypot(d.de_m, d.dn_m)
    d = d[(d.err_m < MAX_ERR_M) & np.isfinite(d.gt_heading)]
    phi = np.radians(d.gt_heading)
    d["fwd_m"] = d.de_m * np.sin(phi) + d.dn_m * np.cos(phi)
    d["lat_m"] = d.de_m * np.cos(phi) - d.dn_m * np.sin(phi)
    return d


def attach_tilt(d: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows = []
    for rid in sorted(d.region.unique()):
        csv = next((root / rid).glob("*.csv"))
        fr = _read_csv(csv)[["filename", "Omega", "Kappa"]]
        fr["drone_path"] = f"{rid}/drone/" + fr.filename.astype(str)
        rows.append(fr)
    tilt = pd.concat(rows)[["drone_path", "Omega", "Kappa"]]
    out = d.reset_index().merge(tilt, on="drone_path", how="left").set_index("pair_id")
    return out[np.isfinite(out.Omega) & np.isfinite(out.Kappa)]


def design(d: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(d)), d.Omega, d.Kappa])


def fit(d: pd.DataFrame, per_agl: bool) -> dict:
    X = design(d)
    s = d.agl_m.to_numpy() if per_agl else 1.0
    coef = {}
    for comp in ("lat_m", "fwd_m"):
        y = d[comp].to_numpy() / s
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
        coef[comp] = {"a": float(beta[0]), "b_omega": float(beta[1]), "c_kappa": float(beta[2]),
                      "r2_train": float(r2)}
    return coef


def residual(d: pd.DataFrame, coef: dict, per_agl: bool) -> pd.DataFrame:
    X = design(d)
    s = d.agl_m.to_numpy() if per_agl else 1.0
    r = d.copy()
    for comp in ("lat_m", "fwd_m"):
        c = coef[comp]
        r[comp + "_res"] = d[comp].to_numpy() - (X @ np.array([c["a"], c["b_omega"], c["c_kappa"]])) * s
    r["shift_before"] = np.hypot(d.lat_m, d.fwd_m)
    r["shift_after"] = np.hypot(r.lat_m_res, r.fwd_m_res)
    return r


def reduction_ci(r: pd.DataFrame, B: int, seed: int) -> tuple[float, float]:
    """Medyan kayma azalmasının 50 karelik uçuş blokları üzerinde küme bootstrap %95 GA'sı."""
    blocks = (r.region + "_" + (r.index.str.split("_").str[1].astype(int) // BLOCK).astype(str))
    groups = [np.flatnonzero(blocks.to_numpy() == b) for b in blocks.unique()]
    rng = np.random.default_rng(seed)
    pre, post = r.shift_before.to_numpy(), r.shift_after.to_numpy()
    vals = []
    for _ in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        vals.append(1 - np.median(post[idx]) / np.median(pre[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def summary(r: pd.DataFrame) -> dict:
    pre, post = float(np.median(r.shift_before)), float(np.median(r.shift_after))
    return {"n": len(r), "median_before_m": pre, "median_after_m": post,
            "reduction": 1 - post / pre,
            "median_lat_before_m": float(np.median(r.lat_m)),
            "median_fwd_before_m": float(np.median(r.fwd_m)),
            "corr_lat_omega": float(np.corrcoef(r.lat_m, r.Omega)[0, 1]),
            "corr_fwd_kappa": float(np.corrcoef(r.fwd_m, r.Kappa)[0, 1])}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("cache/v1"))
    ap.add_argument("--data", type=Path, default=Path("data/UAV_VisLoc_dataset"))
    ap.add_argument("--pairs", type=Path, default=Path("data/pairs/pairs_v1.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    sel = yaml.safe_load(Path("configs/baselines.yaml").read_text(encoding="utf-8"))
    px, tau2 = sel["ransac_px"], int(sel["tau_inliers"][2])
    pairs = pd.read_csv(a.pairs, dtype={"region": str, "tile_region": str})
    root = find_root(a.data)
    B, seed = int(cfg["eval"]["bootstrap_B"]), int(cfg["seed"])

    data = {}
    for sp in ("train", "calib"):
        feats = pd.read_csv(OUT / "baselines" / f"feats_{a.cache.name}_{sp}_px{px:g}.csv")
        data[sp] = attach_tilt(selected_positives(feats, pairs[pairs.split == sp], tau2), root)
        print(f"{sp}: {len(data[sp])} seçilmiş pozitif (satır 2 kabul, hata < {MAX_ERR_M:g} m)")

    res = {"rule": f"calib medyan kayma azalması ≥ %{RULE_REDUCTION * 100:.0f} → pairs_v2 (ikincil)",
           "selection": {"row": 2, "ransac_px": px, "tau2": tau2, "max_err_m": MAX_ERR_M},
           "models": {}}
    for name, per_agl in (("metre (kural)", False), ("AGL'ye bölünmüş (ek bilgi)", True)):
        coef = fit(data["train"], per_agl)
        m = {"coef": coef}
        for sp in ("train", "calib"):
            r = residual(data[sp], coef, per_agl)
            m[sp] = summary(r)
            m[sp]["by_region"] = {rid: summary(g) for rid, g in r.groupby("region")}
            if sp == "calib":
                m[sp]["reduction_ci95"] = reduction_ci(r, B, seed)
        res["models"][name] = m

    rule = res["models"]["metre (kural)"]["calib"]
    res["decision"] = ("pairs_v2 (ikincil sonuç)" if rule["reduction"] >= RULE_REDUCTION
                       else "konu kapanır (GT = CSV)")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "k1_tilt.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# K1 — GT kayması ve kamera eğimi (önceden konmuş kural)", "",
             (f"Seçim: satır 2 (RANSAC {px:g} px, τ₂ = {tau2}) kabul + hata < {MAX_ERR_M:g} m. "
             "Model YALNIZCA eğitimde kuruldu. Kayma = RoMa görüntü merkezi − CSV konumu, drone çerçevesinde."),
             "", "| Model | küme | n | medyan kayma önce (m) | sonra (m) | azalma | %95 GA |",
             "|---|---|---|---|---|---|---|"]
    for name, m in res["models"].items():
        for sp in ("train", "calib"):
            s = m[sp]
            ci = s.get("reduction_ci95")
            cis = f"[{ci[0] * 100:.0f}, {ci[1] * 100:.0f}] %" if ci else "–"
            lines.append(f"| {name} | {sp} | {s['n']} | {s['median_before_m']:.1f} | "
                         f"{s['median_after_m']:.1f} | %{s['reduction'] * 100:.1f} | {cis} |")
    lines += ["", "**Katsayılar (kural modeli, eğitim):**", "",
              "| bileşen | a (m) | b·Omega (m/°) | c·Kappa (m/°) | R² (eğitim) |", "|---|---|---|---|---|"]
    for comp, c in res["models"]["metre (kural)"]["coef"].items():
        lines.append(f"| {comp} | {c['a']:.2f} | {c['b_omega']:.2f} | {c['c_kappa']:.2f} | {c['r2_train']:.3f} |")
    lines += ["", "**Bölge kırılımı (kural modeli):**", "",
              "| küme | bölge | n | önce (m) | sonra (m) | azalma | r(yanal, Omega) | r(ileri, Kappa) |",
              "|---|---|---|---|---|---|---|---|"]
    for sp in ("train", "calib"):
        for rid, s in res["models"]["metre (kural)"][sp]["by_region"].items():
            lines.append(f"| {sp} | {rid} | {s['n']} | {s['median_before_m']:.1f} | {s['median_after_m']:.1f} | "
                         f"%{s['reduction'] * 100:.1f} | {s['corr_lat_omega']:.2f} | {s['corr_fwd_kappa']:.2f} |")
    lines += ["", f"**Karar (kural):** calib azalması %{rule['reduction'] * 100:.1f} → **{res['decision']}**."]
    (OUT / "k1_tilt.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    for sp in ("train", "calib"):
        s = res["models"]["metre (kural)"][sp]
        log_run("41_gt_tilt.py", a.config, seed, sp, "k1_median_shift_before_m", round(s["median_before_m"], 3))
        log_run("41_gt_tilt.py", a.config, seed, sp, "k1_median_shift_after_m", round(s["median_after_m"], 3))
    log_run("41_gt_tilt.py", a.config, seed, "calib", "k1_reduction", round(rule["reduction"], 4),
            notes=f"karar={res['decision']} n_train={data['train'].shape[0]} cache={a.cache.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
