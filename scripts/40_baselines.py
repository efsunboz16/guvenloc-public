"""Faz 1F – taban çizgileri (ablasyon satırları 1–3), önbellekten.

  # 1) özellikler (CPU, paralel): her (çift, yön) için homografi + inlier + entropi
  python scripts/40_baselines.py feats --cache cache/dev_base --split train --ransac-px 3 5 8
  python scripts/40_baselines.py feats --cache cache/dev_base --split calib --ransac-px 5
  # 2) eşik seçimi YALNIZCA eğitim kümesinde → configs/baselines.yaml
  python scripts/40_baselines.py select --cache cache/dev_base
  # 3) değerlendirme (calib) → results/faz1/baselines/*.md|csv + results/runs.csv
  python scripts/40_baselines.py eval --cache cache/dev_base --split calib

Seçim kuralı: her satır için τ = eğitimde FAR-neg ≤ α (= 0,05) veren en küçük inlier eşiği;
RANSAC eşiği adaylar içinden satır 2'nin eğitim başarı@25'ini en yüksek yapan.
Güven aralıkları: 50 ardışık karelik uçuş blokları üzerinde küme bootstrap (calib'de yalnızca
2 uçuş olduğundan uçuş bazlı bootstrap anlamsız).
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc import metrics as M
from guvenloc import stats as S
from guvenloc.baselines import ROWS, features, select_rotation
from guvenloc.cache import pair_file
from guvenloc.geo import angular_error_deg
from guvenloc.matcher import MatchSet

OUT = Path("results/faz1/baselines")
ROTS = (0, 90, 180, 270)
BLOCK = 50


def feats_path(cache: Path, split: str, px: float) -> Path:
    return OUT / f"feats_{cache.name}_{split}_px{px:g}.csv"


def _job(args):
    path, pair_id, rot, pxs, seed = args
    if not path.exists():
        return []
    ms = MatchSet.load_npz(path)
    return [{"pair_id": pair_id, "rot": rot, "ransac_px": px,
             **features(ms, ransac_px=px, seed=seed)} for px in pxs]


def cmd_feats(a, cfg, pairs):
    df = pairs[pairs.split == a.split]
    jobs = [(pair_file(a.cache, r.pair_id, rot), r.pair_id, rot, a.ransac_px, cfg["seed"])
            for r in df.itertuples() for rot in ROTS]
    jobs = [j for j in jobs if j[0].exists()]
    print(f"{a.split}: {len(jobs)} önbellek dosyası ({df.pair_id.nunique()} çift)")
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(_job, jobs, chunksize=32), 1):
            rows += r
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    f = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    for px, g in f.groupby("ransac_px"):
        g.drop(columns="ransac_px").to_csv(feats_path(a.cache, a.split, px), index=False)
        print("yazıldı:", feats_path(a.cache, a.split, px))


def evaluate_rows(feats: pd.DataFrame, pairs: pd.DataFrame, taus: dict | None):
    """Her çift için satır 1–3 kararları. Dönüş: uzun tablo (row, pair_id, ...)."""
    full = feats.groupby("pair_id").filter(lambda g: set(g.rot) >= set(ROTS))
    meta = pairs.set_index("pair_id")
    tile_mpp = (meta.tile_size_m / meta.tile_px)
    out = []
    for pid, g in full.groupby("pair_id"):
        m = meta.loc[pid]
        for row in (1, 2, 3):
            s = select_rotation(g, row)
            err = (np.hypot(s.cx - m.gt_x, s.cy - m.gt_y) * tile_mpp[pid]
                   if m.label == "pos" and s.valid and s.n_inliers > 0 else np.nan)
            herr = (angular_error_deg(s.bearing, m.gt_heading)
                    if m.label == "pos" and s.valid and np.isfinite(m.gt_heading) else np.nan)
            out.append({"row": row, "pair_id": pid, "rot": s.rot, "score": s.n_inliers,
                        "err_m": err, "heading_err": herr, "is_pos": m.label == "pos",
                        "neg_type": m.neg_type, "region": m.region, "terrain": m.terrain,
                        "gt_heading": m.gt_heading, "agl_m": m.agl_m,
                        "block": f"{m.region}_{int(pid.split('_')[1]) // BLOCK}"})
    r = pd.DataFrame(out)
    if taus is not None:
        r["accept"] = r.score >= r.row.map(taus)
    return r


def tau_for_far(scores_neg: np.ndarray, alpha: float) -> int:
    """FAR-neg ≤ α veren en küçük tamsayı inlier eşiği."""
    s = np.sort(np.asarray(scores_neg))
    for t in range(int(s.max()) + 2 if len(s) else 1):
        if (s >= t).mean() <= alpha:
            return t
    return int(s.max()) + 1


def cmd_select(a, cfg, pairs):
    alpha = float(cfg["eval"].get("alpha_far", 0.05))
    best = None
    rep = []
    for px in a.ransac_px:
        p = feats_path(a.cache, "train", px)
        if not p.exists():
            print("yok:", p)
            continue
        r = evaluate_rows(pd.read_csv(p), pairs, None)
        taus = {row: tau_for_far(g[~g.is_pos].score.to_numpy(), alpha)
                for row, g in r.groupby("row")}
        r["accept"] = r.score >= r.row.map(taus)
        for row, g in r.groupby("row"):
            pos = g[g.is_pos]
            s25 = M.success_at(pos.err_m, 25, pos.accept)
            rep.append({"ransac_px": px, "row": row, "tau": taus[row], "success@25": s25,
                        "far_neg": M.far_neg(g.accept, g.is_pos), "n_pairs": len(g)})
        s2 = next(x["success@25"] for x in rep if x["ransac_px"] == px and x["row"] == 2)
        if best is None or s2 > best[1]:
            best = (px, s2, taus)
    print(pd.DataFrame(rep).round(4).to_string(index=False))
    px, s2, taus = best
    sel = {"selected_on": "train (eğitim alt kümesi; önbellekte bulunan çiftler)",
           "cache": a.cache.name, "alpha_far_neg": alpha, "ransac_px": float(px),
           "tau_inliers": {int(k): int(v) for k, v in taus.items()},
           "train_success@25_row2": round(float(s2), 4),
           "candidates": [{k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                           for k, v in x.items()} for x in rep]}
    Path("configs/baselines.yaml").write_text(
        "# Faz 1F taban çizgisi eşikleri — scripts/40_baselines.py select ile ÜRETİLDİ.\n"
        "# Yalnızca eğitim kümesinde seçildi; calib/test'e dokunulmadı.\n"
        + yaml.safe_dump(sel, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"seçilen: RANSAC {px} px, τ = {taus} → configs/baselines.yaml")


def summarize(g: pd.DataFrame, B: int, seed: int) -> dict:
    pos, k = g[g.is_pos], [10, 25, 50]
    acc_pos = pos.accept.to_numpy()
    good = M.good_label(g.is_pos, g.err_m)
    d = {f"success@{t}": M.success_at(pos.err_m, t, acc_pos) for t in k}
    d.update({f"success@{t}_all": M.success_at(pos.err_m, t) for t in k})
    d["median_err_m_accepted"] = M.median_error(pos.err_m, acc_pos)
    d["median_err_m_all"] = M.median_error(pos.err_m)
    d["median_heading_err"] = float(np.nanmedian(pos.heading_err)) if pos.heading_err.notna().any() else np.nan
    d["auroc"] = M.auroc(g.score, good)
    d["far_neg"] = M.far_neg(g.accept, g.is_pos)
    d["far_neg_hard"] = M.far_neg(g.accept[g.neg_type != "easy"], g.is_pos[g.neg_type != "easy"])
    d["far_neg_easy"] = M.far_neg(g.accept[g.neg_type != "hard"], g.is_pos[g.neg_type != "hard"])
    d["far_loc"] = M.far_loc(g.accept, g.is_pos, g.err_m)
    d["n_pos"], d["n_neg"] = int(g.is_pos.sum()), int((~g.is_pos).sum())
    succ = (acc_pos & (pos.err_m.to_numpy() <= 25))
    _, lo, hi = S.cluster_bootstrap_ci(lambda i: succ[i].mean(), pos.block.to_numpy(), B=B, seed=seed)
    d["success@25_ci"] = [lo, hi]
    sc, gl, bl = g.score.to_numpy(), good, g.block.to_numpy()
    _, lo, hi = S.cluster_bootstrap_ci(lambda i: M.auroc(sc[i], gl[i]), bl, B=B, seed=seed)
    d["auroc_ci"] = [lo, hi]
    return d


def cmd_eval(a, cfg, pairs):
    sel = yaml.safe_load(Path("configs/baselines.yaml").read_text(encoding="utf-8"))
    px, taus = sel["ransac_px"], {int(k): v for k, v in sel["tau_inliers"].items()}
    feats = pd.read_csv(feats_path(a.cache, a.split, px))
    r = evaluate_rows(feats, pairs, taus)
    B = int(cfg["eval"]["bootstrap_B"]) if not a.quick else 200
    seed = int(cfg["seed"])
    calls = {1: 1, 2: 4, 3: 4}
    summ = {row: {**summarize(g, B, seed), "calls_per_frame": calls[row]}
            for row, g in r.groupby("row")}

    # eşleştirilmiş karşılaştırmalar (pozitifler, başarı@25): 2 vs 1, 3 vs 1, 3 vs 2
    piv = r[r.is_pos].assign(ok=lambda x: x.accept & (x.err_m <= 25)).pivot(
        index="pair_id", columns="row", values="ok")
    comps = [(2, 1), (3, 1), (3, 2)]
    tests = [S.mcnemar(piv[i].to_numpy(bool), piv[j].to_numpy(bool)) for i, j in comps]
    padj = S.holm([t["p"] for t in tests])

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{a.cache.name}_{a.split}"
    r.to_csv(OUT / f"decisions_{tag}.csv", index=False)
    lines = [f"# Taban çizgileri (satır 1–3) — {a.split}, önbellek `{a.cache.name}`", "",
             (f"RANSAC {px:g} px (karo pikseli), τ (inlier) = {taus} — eğitimde FAR-neg ≤ "
              f"{sel['alpha_far_neg']} ile seçildi. Pozitif {summ[1]['n_pos']}, negatif "
              f"{summ[1]['n_neg']} çift. GA: %95, {B} tekrar, {BLOCK} karelik uçuş blokları."), "",
             ("| Satır | Yöntem | çağrı/kare | başarı@10 | başarı@25 [GA] | başarı@50 "
              "| başarı@25 (retsiz) | medyan hata (m, kabul) | medyan yön hatası | AUROC [GA] "
              "| FAR-neg (zor/kolay) | FAR-loc |"),
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in (1, 2, 3):
        d = summ[row]
        lines.append(
            f"| {row} | {ROWS[row]} | {d['calls_per_frame']} | {d['success@10']:.3f} | "
            f"{d['success@25']:.3f} [{d['success@25_ci'][0]:.3f}, {d['success@25_ci'][1]:.3f}] | "
            f"{d['success@50']:.3f} | {d['success@25_all']:.3f} | {d['median_err_m_accepted']:.1f} | "
            f"{d['median_heading_err']:.1f}° | {d['auroc']:.3f} [{d['auroc_ci'][0]:.3f}, "
            f"{d['auroc_ci'][1]:.3f}] | {d['far_neg']:.3f} ({d['far_neg_hard']:.3f}/"
            f"{d['far_neg_easy']:.3f}) | {d['far_loc']:.3f} |")
    lines += ["", "Eşleştirilmiş McNemar (başarı@25, pozitifler), Holm düzeltmeli:", "",
              "| Karşılaştırma | yalnız ilki doğru | yalnız ikincisi doğru | p | p (Holm) |",
              "|---|---|---|---|---|"]
    for (i, j), t, pa in zip(comps, tests, padj, strict=True):
        lines.append(f"| {i} vs {j} | {t['a_only']} | {t['b_only']} | {t['p']:.2e} | {pa:.2e} |")

    def breakdown(col, name, fn=None):
        rr = r[r.is_pos].copy()
        rr["k"] = fn(rr) if fn else rr[col]
        rows_ = ["", f"Kırılım — {name} (başarı@25, kabul edilmiş; n):", "",
                 "| " + name + " | n | satır 1 | satır 2 | satır 3 |", "|---|---|---|---|---|"]
        for k, g in rr.groupby("k"):
            v = [M.success_at(g[g.row == row].err_m, 25, g[g.row == row].accept) for row in (1, 2, 3)]
            rows_.append(f"| {k} | {len(g) // 3} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
        return rows_

    lines += breakdown(None, "yön dilimi (°)",
                       lambda x: (M.heading_bin(x.gt_heading) * 45).astype(str) + "–"
                       + (M.heading_bin(x.gt_heading) * 45 + 45).astype(str))
    lines += breakdown(None, "AGL irtifa (m)", lambda x: pd.cut(
        x.agl_m, [0, 400, 800, 1200, 2000, 5000],
        labels=["<400", "400–800", "800–1200", "1200–2000", ">2000"]).astype(str))
    lines += breakdown("terrain", "arazi")
    lines += breakdown("region", "bölge")
    md = OUT / f"table_{tag}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))
    print("tablo:", md)

    for row, d in summ.items():
        for k in ("success@10", "success@25", "success@50", "success@25_all", "median_err_m_accepted",
                  "auroc", "far_neg", "far_loc", "median_heading_err"):
            M.log_run("40_baselines.py", Path("configs/baselines.yaml"), seed, a.split,
                      f"row{row}_{k}", round(float(d[k]), 5), notes=f"cache={a.cache.name} px={px:g}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["feats", "select", "eval"])
    ap.add_argument("--cache", type=Path, default=Path("cache/dev_base"))
    ap.add_argument("--split", default="calib")
    ap.add_argument("--ransac-px", type=float, nargs="+", default=[3.0, 5.0, 8.0])
    ap.add_argument("--pairs", type=Path, default=Path("data/pairs/pairs_v1.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--quick", action="store_true", help="bootstrap B = 200")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    pairs = pd.read_csv(a.pairs, dtype={"region": str, "tile_region": str})
    {"feats": cmd_feats, "select": cmd_select, "eval": cmd_eval}[a.cmd](a, cfg, pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
