"""Faz 3B–3C – ret skoru S (lojistik, eğitim) ve H2 kuru denemesi (calib), `configs/h2.yaml`.

  python scripts/61_m2_score.py                   # calib; M1-C dönmeleri v0.1 modelinden (F2-9)

Satırlar (aynı RoMa çağrıları):
  4a   M1-C en olası dönme; konum RANSAC, skor inlier, τ = 709 (eğitim FAR-neg ≤ 0,05; Faz 2 ile aynı)
  5a   aynı dosya; konum hacim tepesi, skor S, τ_S calib'de başarı@25(4a)'ya eşlenir (h2.yaml)
  5a-x tek metrik skorlar (destek, ρ, H) — ablasyon alt satırı, aynı eşik eşleme
  2    4 dönme, en çok inlier, τ = 1.462 (Faz 1)      |  2v  4 dönme, en yüksek S, τ eşlenir (ikincil)
Doğru = pozitif ∧ hata ≤ 25 m (kendi konumuyla); yanlış kabul = kabul ∧ ¬doğru, tüm çiftler üzerinden.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc import metrics as M
from guvenloc import stats as S
from guvenloc.metrics import log_run

OUT = Path("results/faz3/m2")
BLOCK = 50
TAU_4A = 709.0      # Faz 2 satır 4a eşiği: eğitim negatifleri, 4 dönme havuzu, FAR-neg ≤ 0,05 (56_m1c_roma.py)
M1_DECISIONS = "results/faz2/m1/roma_row4_decisions_D2_olcek_rot90_s2_calib.csv"


def feat_matrix(df: pd.DataFrame, names) -> np.ndarray:
    cols = {"log1p_support": np.log1p(df.support.clip(lower=0)), "log1p_sum_c": np.log1p(df.sum_c),
            "support_frac": df.support_frac, "rho": df.rho, "spread_m": df.spread_m.fillna(50.0),
            "entropy_bits": df.entropy_bits.fillna(df.entropy_bits.max()), "frac_dropped": df.frac_dropped}
    return np.column_stack([np.asarray(cols[n], float) for n in names])


def fit_logistic(X: np.ndarray, y: np.ndarray, C: float, iters: int = 50):
    """L2 lojistik (sklearn kuralı: ½‖w‖² + C Σ logloss; kesişim cezasız), Newton–IRLS."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    reg = np.r_[0.0, np.ones(Z.shape[1] - 1)] / C
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) + reg * w
        Hm = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(reg)
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return {"w": w, "mu": mu, "sd": sd}


def predict(model, X):
    Z = np.column_stack([np.ones(len(X)), (X - model["mu"]) / model["sd"]])
    return Z @ model["w"]                                        # logit (sıralama için yeterli)


def match_tau(score, correct, is_pos, target: float) -> float:
    """başarı@25 ≥ target olan en büyük eşik (calib)."""
    order = np.sort(np.unique(score))[::-1]
    pos = is_pos.sum()
    for t in order:
        if ((score >= t) & correct & is_pos).sum() / pos >= target:
            return float(t)
    return float(order[-1])


def row_stats(score, correct, is_pos, tau, blocks, B, seed) -> dict:
    acc = score >= tau
    fa = acc & ~correct
    d = {"tau": tau, "success@25": float((acc & correct)[is_pos].mean()), "false_accept": float(fa.mean()),
         "far_neg": float(acc[~is_pos].mean()), "far_loc": float((acc & ~correct)[is_pos].mean()),
         "auroc": M.auroc(score, correct), "accept_rate": float(acc.mean())}
    _, lo, hi = S.cluster_bootstrap_ci(lambda i: M.auroc(score[i], correct[i]), blocks, B=B, seed=seed)
    d["auroc_ci"] = [lo, hi]
    return d


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    h2 = yaml.safe_load(Path("configs/h2.yaml").read_text(encoding="utf-8"))
    base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    sel = yaml.safe_load(Path("configs/baselines.yaml").read_text(encoding="utf-8"))
    B, seed = int(h2["bootstrap"]["B"]), int(base["seed"])
    pairs = pd.read_csv("data/pairs/pairs_v1.csv", dtype={"region": str}).set_index("pair_id")
    feats = h2["score_model"]["features"]

    def load(split):
        v = pd.read_csv(OUT / f"vol_feats_v1_{split}.csv")
        r = pd.read_csv(f"results/faz1/baselines/feats_v1_{split}_px5.csv")[["pair_id", "rot", "n_inliers", "cx", "cy", "valid"]]
        d = v.merge(r, on=["pair_id", "rot"]).join(pairs[["label", "gt_x", "gt_y", "tile_size_m", "tile_px", "region"]],
                                                    on="pair_id")
        d["is_pos"] = d.label == "pos"
        d["ok_m2"] = M.good_label(d.is_pos, d.err_m)
        mpp = d.tile_size_m / d.tile_px
        e_r = np.where(d.valid & (d.n_inliers > 0), np.hypot(d.cx - d.gt_x, d.cy - d.gt_y) * mpp, np.inf)
        d["err_ransac"] = np.where(d.is_pos, e_r, np.nan)
        d["ok_ransac"] = M.good_label(d.is_pos, d.err_ransac)
        d["block"] = d.region + "_" + (d.pair_id.str.split("_").str[1].astype(int) // BLOCK).astype(str)
        return d

    tr, ca = load("train"), load("calib")
    model = fit_logistic(feat_matrix(tr, feats), tr.ok_m2.to_numpy(float), float(h2["score_model"]["C"]))
    tr["S"] = predict(model, feat_matrix(tr, feats))
    ca["S"] = predict(model, feat_matrix(ca, feats))
    singles = {n: (lambda df, n=n: feat_matrix(df, [n])[:, 0] * (-1 if n in ("rho", "entropy_bits") else 1))
               for n in h2["ablation_single"]}
    for n, f in singles.items():
        ca[f"S_{n}"] = f(ca)
    print(f"S eğitim AUROC (tüm dönmeler, örneklem içi): {M.auroc(tr.S, tr.ok_m2):.3f}; "
          f"inlier eğitim AUROC: {M.auroc(tr.n_inliers, tr.ok_ransac):.3f}")

    # 4a/5a: M1-C'nin en olası dönmesi (Faz 2, v0.1 modeli)
    dec = pd.read_csv(M1_DECISIONS)
    rot_4a = dec[dec.row == "4a"].set_index("pair_id").rot
    a4 = ca.set_index(["pair_id", "rot"]).loc[list(zip(rot_4a.index, rot_4a.to_numpy(), strict=True))].reset_index()
    tau4 = TAU_4A
    blocks = a4.block.to_numpy()
    isp = a4.is_pos.to_numpy()
    res = {"4a": row_stats(a4.n_inliers.to_numpy(float), a4.ok_ransac.to_numpy(), isp, tau4, blocks, B, seed)}
    tgt = res["4a"]["success@25"]
    for key, col in [("5a", "S")] + [(f"5a-{n}", f"S_{n}") for n in singles]:
        sc = a4[col].to_numpy(float)
        res[key] = row_stats(sc, a4.ok_m2.to_numpy(), isp, match_tau(sc, a4.ok_m2.to_numpy(), isp, tgt),
                             blocks, B, seed)
    fa4 = (a4.n_inliers.to_numpy() >= tau4) & ~a4.ok_ransac.to_numpy()
    fa5 = (a4.S.to_numpy() >= res["5a"]["tau"]) & ~a4.ok_m2.to_numpy()
    mc = S.mcnemar(~fa5, ~fa4)              # a_only: 5a yanlış kabul etmedi, 4a etti
    ok4, ok5 = a4.ok_ransac.to_numpy(), a4.ok_m2.to_numpy()
    sc4, sc5 = a4.n_inliers.to_numpy(float), a4.S.to_numpy(float)
    dauc = S.cluster_bootstrap_ci(lambda i: M.auroc(sc5[i], ok5[i]) - M.auroc(sc4[i], ok4[i]), blocks, B=B, seed=seed)

    # 2 / 2v: 4 dönme
    g = ca.sort_values("pair_id")
    best_i = g.loc[g.groupby("pair_id").n_inliers.idxmax()]
    best_s = g.loc[g.groupby("pair_id").S.idxmax()]
    b2 = best_i.block.to_numpy()
    res["2"] = row_stats(best_i.n_inliers.to_numpy(float), best_i.ok_ransac.to_numpy(), best_i.is_pos.to_numpy(),
                         float(sel["tau_inliers"][2]), b2, B, seed)
    sc = best_s.S.to_numpy(float)
    res["2v"] = row_stats(sc, best_s.ok_m2.to_numpy(), best_s.is_pos.to_numpy(),
                          match_tau(sc, best_s.ok_m2.to_numpy(), best_s.is_pos.to_numpy(), res["2"]["success@25"]),
                          best_s.block.to_numpy(), B, seed)
    fa2 = (best_i.n_inliers.to_numpy() >= res["2"]["tau"]) & ~best_i.ok_ransac.to_numpy()
    fa2v = (sc >= res["2v"]["tau"]) & ~best_s.ok_m2.to_numpy()
    mc2 = S.mcnemar(~fa2v, ~fa2)

    crit = h2["criteria"]
    h2res = {"far_p": mc["p"], "far_5a_lower": res["5a"]["false_accept"] < res["4a"]["false_accept"],
             "dauc": list(dauc)}
    h2res["i"] = bool(h2res["far_5a_lower"] and mc["p"] < crit["far_mcnemar_alpha"])
    h2res["ii"] = bool(dauc[1] > crit["auroc_diff_ci_low"])
    h2res["H2"] = h2res["i"] and h2res["ii"]

    names = {"4a": "M1-C 1 çağrı · RANSAC + inlier", "5a": "M1-C 1 çağrı · hacim + S",
             **{f"5a-{n}": f"  hacim, yalnız {n}" for n in singles},
             "2": "4 yön · RANSAC + inlier", "2v": "4 yön · hacim + S (en yüksek S)"}
    lines = ["# Faz 3 — H2 kuru denemesi (calib 05, 11), `configs/h2.yaml`", "",
             (f"S: lojistik, eğitim {len(tr)} (çift, dönme); öznitelikler {feats}. Eşik eşleme: 5a/5a-x başarı@25'i "
             f"4a'ya ({tgt:.3f}), 2v satır 2'ye. Doğru = pozitif ∧ hata ≤ 25 m; yanlış kabul = kabul ∧ ¬doğru."), "",
             "| satır | yöntem | τ | başarı@25 | yanlış kabul (tüm) | FAR-neg | FAR-loc | AUROC [GA] |",
             "|---|---|---|---|---|---|---|---|"]
    for k, nm in names.items():
        d = res[k]
        lines.append(f"| {k} | {nm} | {d['tau']:.3g} | {d['success@25']:.3f} | {d['false_accept']:.3f} | "
                     f"{d['far_neg']:.3f} | {d['far_loc']:.3f} | {d['auroc']:.3f} [{d['auroc_ci'][0]:.3f}, "
                     f"{d['auroc_ci'][1]:.3f}] |")
    lines += ["", (f"Yanlış kabul McNemar (5a vs 4a): yalnız 4a yanlış kabul {mc['a_only']}, yalnız 5a {mc['b_only']}, "
              f"p = {mc['p']:.2e}. 2v vs 2: yalnız 2 {mc2['a_only']}, yalnız 2v {mc2['b_only']}, p = {mc2['p']:.2e}."),
              f"AUROC farkı (5a − 4a): {dauc[0]:+.3f} [{dauc[1]:+.3f}, {dauc[2]:+.3f}].", "",
              (f"H2 (`configs/h2.yaml`): (i) yanlış kabul 5a < 4a ve p < {crit['far_mcnemar_alpha']} → "
              f"{'✓' if h2res['i'] else '✗'}; (ii) AUROC farkı GA alt sınırı > 0 → {'✓' if h2res['ii'] else '✗'}. "
              f"**H2 {'sağlandı' if h2res['H2'] else 'sağlanmadı'}** (calib; karar testte verilir)."), "",
              "S katsayıları (standartlaştırılmış): " + ", ".join(
                  f"{n} {w:+.2f}" for n, w in zip(["kesişim", *feats], model["w"], strict=True))]
    md = OUT / "h2_calib.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "h2_calib.json").write_text(json.dumps(
        {"rows": res, "mcnemar_5a_4a": mc, "mcnemar_2v_2": mc2, "auroc_diff": list(dauc), "h2": h2res,
         "model": {k: np.asarray(v).tolist() for k, v in model.items()}, "features": feats},
        indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("\n".join(lines))
    for k in ("4a", "5a", "2", "2v"):
        for m in ("success@25", "false_accept", "auroc"):
            log_run("61_m2_score.py", Path("configs/h2.yaml"), seed, "calib", f"row{k}_{m}", round(res[k][m], 5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
