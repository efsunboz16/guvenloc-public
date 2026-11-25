"""Faz 1 §4c – RoMa v2 belirsizlik ölçeği `sigma_scale` (Σ ← s² · Σ) kalibrasyonu.

  python scripts/42_sigma_scale.py --cache cache/v1            # hesapla + rapor
  python scripts/42_sigma_scale.py --cache cache/v1 --write    # + configs/base.yaml'a yaz

Yöntem (faz1_kalan_isler.md §4c, önceden konmuş):
  * Eğitim pozitiflerinde satır 2'nin (4 yön + en çok inlier) seçtiği ve KABUL ettiği, hatası < 25 m olan
    çiftler. O yönün homografisi yeniden kurulur (40_baselines ile aynı RANSAC eşiği ve tohum).
  * RANSAC inlier'ları için artık r_i = v_i − H(u_i) (karo pikseli), m_i = r_iᵀ Σ_B,i⁻¹ r_i.
  * Gauss NLL'yi en küçükleyen ölçek: s² = mean(m_i) / 2  (2 boyut).
  * Kontrol: ölçeklenmiş %95 elipslerinin (m_i / s² ≤ χ²₂(0,95) = 5,991) kapsaması eğitimde ≈ %95;
    calib'de aynı seçimle raporlanır.
Ek bilgi (karar için kullanılmaz): sağlam (medyan) ve kapsama eşleştiren ölçekler, bölge kırılımı.

Not: `base` önbelleği tek yönlüdür (Σ_A = 0), dolayısıyla artığın kovaryansı yalnızca Σ_B'dir.
Önbellek ham Σ'yı saklar (meta.sigma_scale = 1,0); tüketiciler s = config / meta oranını uygular.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc.baselines import fit_homography, project
from guvenloc.cache import pair_file
from guvenloc.matcher import MatchSet
from guvenloc.metrics import log_run

OUT = Path("results/faz1")
MAX_ERR_M = 25.0
CHI2_2_95 = 5.991464547107979
CHI2_2_MEDIAN = 2 * np.log(2)


def _baselines_module():
    spec = importlib.util.spec_from_file_location("b40", Path(__file__).with_name("40_baselines.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mahalanobis_inliers(args) -> tuple[str, str, np.ndarray]:
    path, pair_id, region, px, seed = args
    ms = MatchSet.load_npz(path)
    if float(ms.meta.get("sigma_scale", 1.0)) != 1.0:
        raise ValueError(f"{path}: ham Σ bekleniyordu (meta.sigma_scale = 1)")
    fit = fit_homography(ms, ransac_px=px, seed=seed)
    if fit["H"] is None:
        return pair_id, region, np.empty(0, np.float32)
    inl = fit["inliers"]
    r = (ms.v[inl].astype(np.float64) - project(fit["H"], ms.u[inl].astype(np.float64)))
    S = ms.Sigma_B[inl].astype(np.float64)
    det = S[:, 0, 0] * S[:, 1, 1] - S[:, 0, 1] * S[:, 1, 0]
    # 2×2 ters: [[d, −b], [−c, a]] / det
    m = (S[:, 1, 1] * r[:, 0] ** 2 - (S[:, 0, 1] + S[:, 1, 0]) * r[:, 0] * r[:, 1]
         + S[:, 0, 0] * r[:, 1] ** 2) / det
    return pair_id, region, m.astype(np.float32)


def collect(split: str, a, cfg, sel, pairs, b40) -> pd.DataFrame:
    px, taus = sel["ransac_px"], {int(k): v for k, v in sel["tau_inliers"].items()}
    feats = pd.read_csv(OUT / "baselines" / f"feats_{a.cache.name}_{split}_px{px:g}.csv")
    r = b40.evaluate_rows(feats, pairs[pairs.split == split], taus)
    ok = r[(r.row == 2) & r.is_pos & r.accept & (r.err_m < MAX_ERR_M)]
    jobs = [(pair_file(a.cache, x.pair_id, int(x.rot)), x.pair_id, x.region, px, int(cfg["seed"]))
            for x in ok.itertuples()]
    out = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for pid, rid, m in ex.map(mahalanobis_inliers, jobs, chunksize=8):
            out.append(pd.DataFrame({"pair_id": pid, "region": rid, "m": m}))
    d = pd.concat(out, ignore_index=True)
    print(f"{split}: {len(ok)} çift, {len(d)} inlier")
    return d


def stats(m: np.ndarray, s2: float) -> dict:
    return {"n_inliers": len(m), "s2_nll": float(np.mean(m) / 2),
            "s_nll": float(np.sqrt(np.mean(m) / 2)),
            "s_robust_median": float(np.sqrt(np.median(m) / CHI2_2_MEDIAN)),
            "s_cov95": float(np.sqrt(np.quantile(m, 0.95) / CHI2_2_95)),
            "coverage95_at_s": float(np.mean(m / s2 <= CHI2_2_95)),
            "coverage95_raw": float(np.mean(m <= CHI2_2_95)),
            "coverage68_at_s": float(np.mean(m / s2 <= 2.2788)),
            "median_m": float(np.median(m))}


def write_config(path: Path, s: float) -> None:
    txt = path.read_text(encoding="utf-8")
    new, n = re.subn(r"^(\s*sigma_scale:\s*)[0-9.]+(.*)$",
                     lambda mm: f"{mm[1]}{s:.4f}            # Σ ← s²·Σ; 42_sigma_scale.py (eğitim, NLL); "
                                "cache/v1 ham Σ saklar (meta 1,0)",
                     txt, count=1, flags=re.MULTILINE)
    if n != 1:
        raise RuntimeError("configs/base.yaml: sigma_scale satırı bulunamadı")
    path.write_text(new, encoding="utf-8")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("cache/v1"))
    ap.add_argument("--pairs", type=Path, default=Path("data/pairs/pairs_v1.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--write", action="store_true", help="sonucu configs/base.yaml'a yaz")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    sel = yaml.safe_load(Path("configs/baselines.yaml").read_text(encoding="utf-8"))
    pairs = pd.read_csv(a.pairs, dtype={"region": str, "tile_region": str})
    b40 = _baselines_module()

    tr = collect("train", a, cfg, sel, pairs, b40)
    s2 = float(np.mean(tr.m) / 2)           # YALNIZCA eğitim
    s = float(np.sqrt(s2))
    ca = collect("calib", a, cfg, sel, pairs, b40)

    res = {"method": "s² = mean(rᵀ Σ_B⁻¹ r) / 2, eğitim, satır 2 kabul + hata < 25 m, RANSAC inlier'ları",
           "ransac_px": sel["ransac_px"], "sigma_scale": s, "sigma_scale_sq": s2,
           "train": stats(tr.m.to_numpy(), s2), "calib": stats(ca.m.to_numpy(), s2),
           "by_region": {f"{sp}/{rid}": stats(g.m.to_numpy(), s2)
                         for sp, d in (("train", tr), ("calib", ca)) for rid, g in d.groupby("region")},
           "n_pairs": {"train": int(tr.pair_id.nunique()), "calib": int(ca.pair_id.nunique())}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sigma_scale.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# `sigma_scale` kalibrasyonu (Faz 1 §4c)", "",
             (f"Eğitim: {res['n_pairs']['train']} çift, {res['train']['n_inliers']} inlier (satır 2 kabul + hata < 25 m, "
             f"RANSAC {sel['ransac_px']:g} px). **s = {s:.4f}** (s² = {s2:.4f}) — Gauss NLL, yalnızca eğitimde."), "",
             "| küme | bölge | inlier | s (NLL) | s (medyan, ek) | s (%95 kapsama, ek) | %95 kapsama, s ile | %68 kapsama, s ile | %95 kapsama, ham Σ |",
             "|---|---|---|---|---|---|---|---|---|"]
    def row(sp, rid, st):
        return (f"| {sp} | {rid} | {st['n_inliers']} | {st['s_nll']:.3f} | {st['s_robust_median']:.3f} | "
                f"{st['s_cov95']:.3f} | %{st['coverage95_at_s'] * 100:.1f} | %{st['coverage68_at_s'] * 100:.1f} | "
                f"%{st['coverage95_raw'] * 100:.1f} |")
    lines.append(row("train", "hepsi", res["train"]))
    lines.append(row("calib", "hepsi", res["calib"]))
    for k, st in res["by_region"].items():
        sp, rid = k.split("/")
        lines.append(row(sp, rid, st))
    (OUT / "sigma_scale.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    seed = int(cfg["seed"])
    log_run("42_sigma_scale.py", a.config, seed, "train", "sigma_scale", round(s, 4),
            notes=f"n_inl={res['train']['n_inliers']} n_pairs={res['n_pairs']['train']}")
    for sp in ("train", "calib"):
        log_run("42_sigma_scale.py", a.config, seed, sp, "sigma_cov95", round(res[sp]["coverage95_at_s"], 4))
    if a.write:
        write_config(a.config, s)
        print(f"configs/base.yaml: romav2.sigma_scale = {s:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
