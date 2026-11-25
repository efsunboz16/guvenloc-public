"""Faz 1 K2 – karo boyutu deneyi: sabit 1400 m vs uyarlamalı (ayak izi köşegeni + 2 × kaydırma).

YALNIZCA eğitim bölgeleri (bölge başına eşit aralıklı N kare, pozitif çiftler); calib/test'e
dokunulmaz. Pozitif karoların merkezleri iki seçenekte aynıdır (aynı RNG) → eşleştirilmiş karşılaştırma.

Önceden konan karar kuralı (docs/faz1_durum.md K2): satır 2'nin (4 yön + en çok inlier) RETSİZ
başarı@25'i eğitimde yüksek olan seçilir; eşitlikte sabit (mevcut pairs_v1) kalır.
Ret eşiği τ iki seçenekte ayrı ayrı seçileceği için karşılaştırma retsiz yapılır (eşleşme kalitesi).

Hazırlık (bkz. docs/faz1_kalan_isler.md §0):
  python scripts/14_make_pairs.py ROOT --set tile_mode=adaptive version=pairs_k2_adaptive
  python scripts/30_build_cache.py ROOT --pairs data/pairs/pairs_k2_adaptive.csv --split train \
      --per-region 60 --neg-types pos --cache cache/k2_adaptive
  (sabit seçenek: cache/dev_base zaten aynı 300 karenin bütün çiftlerini içeriyor)
Kullanım:
  python scripts/15_tile_experiment.py
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc import metrics as M
from guvenloc import stats as S
from guvenloc.baselines import features, select_rotation
from guvenloc.cache import pair_file
from guvenloc.matcher import MatchSet
from guvenloc.metrics import log_run
from guvenloc.pairs import select_per_region

ROTS = (0, 90, 180, 270)
BLOCK = 50
OUT = Path("results/faz1/k2")


def _job(args):
    path, pair_id, rot, px, seed = args
    ms = MatchSet.load_npz(path)
    return {"pair_id": pair_id, "rot": rot, **features(ms, ransac_px=px, seed=seed)}


def run_variant(name, pairs_csv, cache, n_per_region, px, seed, workers):
    df = pd.read_csv(pairs_csv, dtype={"region": str, "tile_region": str})
    df = select_per_region(df[df.split == "train"], n_per_region)
    df = df[df.neg_type == "pos"]
    jobs = [(pair_file(cache, r.pair_id, rot), r.pair_id, rot, px, seed)
            for r in df.itertuples() for rot in ROTS]
    missing = [j[0] for j in jobs if not j[0].exists()]
    if missing:
        raise FileNotFoundError(f"{name}: {len(missing)} önbellek dosyası eksik, ör. {missing[0]}")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        feats = pd.DataFrame(list(ex.map(_job, jobs, chunksize=16)))
    meta = df.set_index("pair_id")
    rows = []
    for pid, g in feats.groupby("pair_id"):
        m = meta.loc[pid]
        mpp = m.tile_size_m / m.tile_px
        for row in (1, 2):
            s = select_rotation(g, row)
            ok = s is not None and s.valid and s.n_inliers > 0
            rows.append({"variant": name, "row": row, "pair_id": pid, "region": m.region,
                         "tile_size_m": m.tile_size_m,
                         "err_m": np.hypot(s.cx - m.gt_x, s.cy - m.gt_y) * mpp if ok else np.nan,
                         "n_inliers": s.n_inliers if s is not None else 0,
                         "block": f"{m.region}_{int(pid.split('_')[1]) // BLOCK}"})
    return pd.DataFrame(rows)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--fixed", nargs=2, type=Path, metavar=("PAIRS", "CACHE"),
                    default=[Path("data/pairs/pairs_v1.csv"), Path("cache/dev_base")])
    ap.add_argument("--adaptive", nargs=2, type=Path, metavar=("PAIRS", "CACHE"),
                    default=[Path("data/pairs/pairs_k2_adaptive.csv"), Path("cache/k2_adaptive")])
    ap.add_argument("--per-region", type=int, default=60)
    ap.add_argument("--ransac-px", type=float, default=None, help="varsayılan: configs/baselines.yaml")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()

    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    px = a.ransac_px or float(yaml.safe_load(
        Path("configs/baselines.yaml").read_text(encoding="utf-8"))["ransac_px"])
    r = pd.concat([run_variant("fixed", *a.fixed, a.per_region, px, seed, a.workers),
                   run_variant("adaptive", *a.adaptive, a.per_region, px, seed, a.workers)])
    OUT.mkdir(parents=True, exist_ok=True)
    r.to_csv(OUT / "decisions.csv", index=False)

    summ = {"ransac_px": px, "per_region": a.per_region, "rule": "row 2 retsiz başarı@25 (eğitim)"}
    lines = ["# K2 karo deneyi — eğitim bölgeleri, pozitif çiftler", "",
             (f"Bölge başına {a.per_region} kare; RANSAC {px:g} px; GA %95, {a.B} tekrar, "
             f"{BLOCK} karelik bloklar. Retsiz (her kare için konum üretilir)."), "",
             "| Satır | Seçenek | karo (m, medyan) | başarı@10 | başarı@25 [GA] | başarı@50 | medyan hata (m) |",
             "|---|---|---|---|---|---|---|"]
    for row in (1, 2):
        piv = {}
        for v in ("fixed", "adaptive"):
            g = r[(r.row == row) & (r.variant == v)].sort_values("pair_id")
            e, blocks = g.err_m.to_numpy(), g.block.to_numpy()
            ok25 = np.isfinite(e) & (e <= 25)
            _, lo, hi = S.cluster_bootstrap_ci(lambda i, o=ok25: o[i].mean(), blocks, B=a.B, seed=seed)
            d = {f"success@{k}": M.success_at(e, k) for k in (10, 25, 50)}
            d.update(success25_ci=[lo, hi], median_err_m=M.median_error(e), n=len(g),
                     tile_m_median=float(g.tile_size_m.median()),
                     by_region={reg: M.success_at(x.err_m, 25) for reg, x in g.groupby("region")})
            summ[f"row{row}_{v}"] = d
            piv[v] = (g.pair_id.to_numpy(), ok25, blocks)
            lines.append(f"| {row} | {v} | {d['tile_m_median']:.0f} | {d['success@10']:.3f} | "
                         f"{d['success@25']:.3f} [{lo:.3f}, {hi:.3f}] | {d['success@50']:.3f} | "
                         f"{d['median_err_m']:.1f} |")
        assert (piv["fixed"][0] == piv["adaptive"][0]).all()
        fa, ad, blocks = piv["fixed"][1], piv["adaptive"][1], piv["fixed"][2]
        mc = S.mcnemar(ad, fa)
        diff, lo, hi = S.cluster_bootstrap_ci(lambda i, x=ad, y=fa: x[i].mean() - y[i].mean(),
                                              blocks, B=a.B, seed=seed)
        summ[f"row{row}_diff"] = {"adaptive_minus_fixed": diff, "ci": [lo, hi], "mcnemar": mc}
        lines.append(f"| {row} | fark (uyarlamalı − sabit) | | | {diff:+.3f} [{lo:+.3f}, {hi:+.3f}] "
                     f"| McNemar {mc['a_only']}'e {mc['b_only']}, p = {mc['p']:.2g} | |")

    s_fix, s_ad = summ["row2_fixed"]["success@25"], summ["row2_adaptive"]["success@25"]
    winner = "adaptive" if s_ad > s_fix else "fixed"
    summ["winner"] = winner
    regions = sorted(summ["row2_fixed"]["by_region"])
    lines += ["", "Bölge başına satır 2 başarı@25 (sabit → uyarlamalı): " + ", ".join(
        f"{reg}: {summ['row2_fixed']['by_region'][reg]:.2f} → {summ['row2_adaptive']['by_region'][reg]:.2f}"
        for reg in regions),
        "", (f"**Kural sonucu: {winner}** (satır 2 retsiz başarı@25: sabit {s_fix:.3f}, "
        f"uyarlamalı {s_ad:.3f}).")]
    (OUT / "k2_tile.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "k2_tile.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False, default=float),
                                      encoding="utf-8")
    print("\n".join(lines))
    for v in ("fixed", "adaptive"):
        log_run("15_tile_experiment.py", a.config, seed, "train", f"k2_{v}_row2_success@25_all",
                summ[f"row2_{v}"]["success@25"], notes=f"per_region={a.per_region} px={px:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
