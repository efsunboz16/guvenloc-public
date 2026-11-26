"""Faz 4 – Modül 3 split conformal: H3 kuru denemeleri (bölge kayması), `configs/h3.yaml`.

  python scripts/64_m3_conformal.py            # dry_runs (05→11, 11→05, 05+11→04) + calib içi tam uyum
  (test bölgeleri 02, 03, 06: Faz 6, tek sefer — bu betik reddeder)

Satır 6 = satır 5a (h2_v3 skoru S, M1-C en olası dönme, hacim tepesi, σ̂ = spread_m) + conformal (τ, q̂).
04 için S, 04 dışarıda bırakılarak yeniden eğitilir; dönme M1-C birleşiminin 04 tahmininden (04 M1-C eğitiminde yok).
→ results/faz4/m3/h3_dry.md / .json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from guvenloc import m3_conformal as C
from guvenloc import stats as ST
from guvenloc.metrics import log_run


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S61 = _load("m2s61", "scripts/61_m2_score.py")
OUT = Path("results/faz4/m3")
TEST = {"02", "03", "06"}
ROTS = (0, 90, 180, 270)


def fit_S(tr: pd.DataFrame, h2: dict):
    sm = h2["score_model"]
    tcol = {"peak": "ok_m2", "shared": "ok_ransac"}[sm.get("target_label", "peak")]
    return S61.fit_logistic(S61.feat_matrix(tr, sm["features"]), tr[tcol].to_numpy(float), float(sm["C"]))


def frame_rows(df: pd.DataFrame, rot: pd.Series) -> pd.DataFrame:
    """(çift, dönme) satırlarından her çift için M1-C'nin seçtiği dönme."""
    return df.set_index(["pair_id", "rot"]).loc[list(zip(rot.index, rot.to_numpy(), strict=True))].reset_index()


def m1_rot_04(model: str, pairs_all: pd.DataFrame) -> pd.Series:
    R56 = _load("m1c56", "scripts/56_m1c_roma.py")
    p = pairs_all[pairs_all.region == "04"].sort_values("pair_id").reset_index(drop=True)
    tiles = R56.tiles_for(p, Path("data/m1/val04_pairs_513.npy"))
    probs, _ = R56.m1_rot_probs(model, p, tiles)
    return pd.Series([ROTS[i] for i in probs.argmax(1)], index=p.pair_id)


def variants(d: pd.DataFrame, floor: float) -> dict:
    """satır → (skor, hata, σ̂ kalibrasyonda, zor dilim ölçütü)."""
    sig = np.maximum(d.spread_m.fillna(50.0).to_numpy(float), floor)
    err_v = np.where(d.is_pos, d.err_m.fillna(np.inf), np.nan)
    err_r = d.err_ransac.to_numpy(float)
    one = np.ones(len(d))
    return {"6": (d.S.to_numpy(float), err_v, sig, sig, None),
            "6-std": (d.S.to_numpy(float), err_v, one, sig, None),
            "6-M": (d.S.to_numpy(float), err_v, sig, sig, 4),
            "6-inl": (d.n_inliers.to_numpy(float), err_r, sig, sig, None)}


def cp(k, n):
    return list(ST.clopper_pearson(int(k), int(n))) if n else [np.nan, np.nan]


def run_pair(fit_d: pd.DataFrame, ev_d: pd.DataFrame, h3: dict) -> dict:
    fl = float(h3["system"]["sigma_floor_m"])
    a, b = float(h3["alpha"]), float(h3["beta"])
    vf, ve = variants(fit_d, fl), variants(ev_d, fl)
    pf, pe = fit_d.is_pos.to_numpy(), ev_d.is_pos.to_numpy()
    out = {}
    for row, (s, e, sg, _hb, mb) in vf.items():
        cal = C.fit(s[~pf], s[pf], e[pf], sg[pf], a, b, mondrian_bins=mb)
        s2, e2, sg2, hb2, _ = ve[row]
        r = C.evaluate(cal, s2, pe, e2, sg2, hard_by=hb2)
        r.update({"tau": cal.tau, "q": cal.q.tolist(), "edges": cal.edges.tolist(),
                  "far_neg_cp": cp(r["k_neg_acc"], r["n_neg"]), "coverage_cp": cp(r["k_cov"], r["n_pos_acc"]),
                  "coverage_hard_cp": cp(r["k_cov_hard"], r["n_hard"])})
        acc = cal.accept(s2)
        r["terrain"] = {t: {"far_neg": float(acc[(~pe) & m].mean()) if ((~pe) & m).sum() else np.nan,
                            "coverage": float((e2 <= cal.radius(sg2))[acc & pe & m].mean())
                            if (acc & pe & m).sum() else np.nan, "n_pos_acc": int((acc & pe & m).sum())}
                        for t in sorted(ev_d.terrain.unique()) for m in [(ev_d.terrain == t).to_numpy()]}
        out[row] = r
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    h3 = yaml.safe_load(Path("configs/h3.yaml").read_text(encoding="utf-8"))
    h2 = yaml.safe_load(Path("configs/h2.yaml").read_text(encoding="utf-8"))
    for dr in h3["dry_runs"]:
        if TEST & set(dr["fit"] + dr["eval"]):
            sys.exit("test bölgeleri Faz 6'da (h3.yaml)")
    pairs_all = pd.read_csv("data/pairs/pairs_v1.csv", dtype={"region": str, "tile_region": str})
    pairs = pairs_all.set_index("pair_id")
    feats = h2["score_model"]["features"]
    model = h3["system"]["m1_model"]

    tr, ca = S61.load_split("train", pairs), S61.load_split("calib", pairs)
    ca["S"] = S61.predict(fit_S(tr, h2), S61.feat_matrix(ca, feats))
    dec = pd.read_csv(S61.M1_DECISIONS.format(model=model))
    ca_f = frame_rows(ca, dec[dec.row == "4a"].set_index("pair_id").rot)
    v04 = tr[tr.region == "04"].copy()
    v04["S"] = S61.predict(fit_S(tr[tr.region != "04"], h2), S61.feat_matrix(v04, feats))
    rot04 = m1_rot_04(model, pairs_all)
    v04_f = frame_rows(v04, rot04)
    data = pd.concat([ca_f, v04_f], ignore_index=True)
    data = data.join(pairs[["terrain", "agl_m"]], on="pair_id")

    res = {}
    for dr in h3["dry_runs"] + [{"fit": ["05", "11"], "eval": ["05", "11"]}]:
        key = "+".join(dr["fit"]) + " → " + "+".join(dr["eval"])
        f, e = data[data.region.isin(dr["fit"])], data[data.region.isin(dr["eval"])]
        res[key] = run_pair(f, e, h3)
        print(f"{key}: " + ", ".join(f"{r} FAR-neg {v['far_neg']:.3f} kapsama {v['coverage']:.3f} "
                                     f"zor {v['coverage_hard']:.3f} yarıçap {v['radius_median_m']:.0f} m"
                                     for r, v in res[key].items()))
    crit = h3["criteria"]
    lines = ["# Faz 4 — H3 kuru denemeleri (bölge kayması; test değil), `configs/h3.yaml`", "",
             (f"Satır 6 = 5a (h2_v3 S, M1-C birleşimi 4a dönmesi, hacim tepesi, σ̂ = spread_m ≥ "
             f"{h3['system']['sigma_floor_m']} m) + split conformal; α = {h3['alpha']}, β = {h3['beta']}. "
             "Kapsama kabul koşullu (pozitif ∧ kabul). Zor çeyrek: kabul edilmiş pozitiflerde σ̂'nin en büyük %25'i. "
             "GA: Clopper–Pearson %95. Son blok (05+11 → 05+11) örneklem içi, yalnız denetim."), ""]
    for key, rows in res.items():
        lines += [f"## {key}", "", ("| satır | FAR-neg [GA] | kabul (poz.) | kapsama [GA] | zor çeyrek [GA] | "
                  "yarıçap medyan / ort. | q̂ | H3 ölçütleri |"), "|---|---|---|---|---|---|---|---|"]
        for row, v in rows.items():
            ok = (v["far_neg"] <= crit["far_neg_max"], v["coverage"] >= crit["coverage_min"],
                  v["coverage_hard"] >= crit["coverage_hard_min"])
            lines.append(f"| {row} | {v['far_neg']:.3f} [{v['far_neg_cp'][0]:.3f}, {v['far_neg_cp'][1]:.3f}] | "
                         f"{v['accept_pos']:.3f} | {v['coverage']:.3f} [{v['coverage_cp'][0]:.3f}, "
                         f"{v['coverage_cp'][1]:.3f}] (n {v['n_pos_acc']}) | {v['coverage_hard']:.3f} "
                         f"[{v['coverage_hard_cp'][0]:.3f}, {v['coverage_hard_cp'][1]:.3f}] | "
                         f"{v['radius_median_m']:.0f} / {v['radius_mean_m']:.0f} m | "
                         + "/".join(f"{q:.2f}" for q in v["q"]) + " | " + " ".join("✓" if o else "✗" for o in ok) + " |")
        lines.append("")
        lines.append("Arazi (satır 6): " + "; ".join(f"{t}: FAR-neg {x['far_neg']:.3f}, kapsama {x['coverage']:.3f} "
                                                     f"(n {x['n_pos_acc']})" for t, x in rows["6"]["terrain"].items()))
        lines.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "h3_dry.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "h3_dry.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("\n".join(lines))
    for key, rows in res.items():
        for m in ("far_neg", "coverage", "coverage_hard"):
            log_run("64_m3_conformal.py", Path("configs/h3.yaml"), 0, key, f"row6_{m}", round(rows["6"][m], 5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
