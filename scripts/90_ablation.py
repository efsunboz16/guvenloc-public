"""Faz 6A – yedi satırlık ablasyon + H1–H4 test kararları (MIMARI 6A). Tek sefer, dondurulmuş manifestle.

  python scripts/90_ablation.py --rehearse   # aynı tablo kodu calib'de (Faz 2–5 çıktılarından), koruyucusuz
  python scripts/90_ablation.py --final      # TEST: 56 (H1) → 61 (H2) → 64 (H3) → 72 (H4) sırayla, sonra tablo

Satırlar (aynı test çiftleri, eşleştirilmiş):
  1 RoMa (0°) + RANSAC · 2 4 yön + en çok inlier · 3 4 yön + en düşük entropi       (eşikler eğitimden)
  4 + M1-C (4e, güvene bağlı bütçe; H1 satırı)
  5 + M2: M1-C 4a dönmesi, konum = hacim tepesi, skor = S, eşik calib'de 4a başarısına eşlenmiş (h2)
  6 + M3: aynı konum ve S, kabul = S > τ (conformal, α = 0,05), yarıçap q̂·σ̂ (h3)
  7 + M4: parçacık filtresi (yalnız pozitif kareler; uçuş dizisi), yarıçap κ·yayılım (h4)
Ölçütler: başarı@10/25/50 (kabul ∧ hata ≤ k, pozitifler), medyan hata (tüm pozitifler, kabulden bağımsız),
AUROC (skor ~ doğru = pozitif ∧ hata ≤ 25 m, satırın kendi konumuyla), FAR-neg, çağrı/kare, kapsama, yarıçap.
Ardışık satırlar: McNemar (başarı@25) ve Wilcoxon (hata), Holm; GA'lar 50 karelik blok bootstrap.
Kırılımlar: arazi, irtifa (AGL üçte birlikleri, eğitim+calib sınırları), yön (en yakın ana yön), bölge.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from guvenloc import metrics as M
from guvenloc import stats as ST


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ROWS = {"1": "RoMa v2 (0°) + RANSAC", "2": "4 yön + en çok inlier", "3": "4 yön + en düşük entropi",
        "4": "+ M1-C (4e)", "5": "+ M2 (hacim tepesi, S)", "6": "+ M3 (conformal)", "7": "+ M4 (filtre)"}
PAIRS_CMP = [("2", "1"), ("3", "2"), ("4", "2"), ("5", "4"), ("6", "5"), ("7", "6")]


def build(split: str, src: Path, h2v: dict, h3p: dict, filt: pd.DataFrame, model: str) -> pd.DataFrame:
    """Uzun karar tablosu: row, pair_id, is_pos, accept, score, err_m, calls, block, terrain, agl_m, gt_heading, region."""
    pairs = pd.read_csv("data/pairs/pairs_v1.csv", dtype={"region": str, "tile_region": str}).set_index("pair_id")
    dec = pd.read_csv(src / f"roma_row4_decisions_{model}_{split}.csv", dtype={"region": str, "row": str})
    keep = ["pair_id", "is_pos", "accept", "score", "err_m", "calls", "block", "terrain", "agl_m", "gt_heading", "region"]
    out = [dec[dec.row == r][keep].assign(row=r) for r in ("1", "2", "3")]
    out.append(dec[dec.row == "4e"][keep].assign(row="4"))
    S61 = _load("m2s61", "scripts/61_m2_score.py")
    S64 = _load("m3s64", "scripts/64_m3_conformal.py")
    h2 = yaml.safe_load(Path("configs/h2.yaml").read_text(encoding="utf-8"))
    tr, ev = S61.load_split("train", pairs), S61.load_split(split, pairs)
    ev["S"] = S61.predict(S64.fit_S(tr, h2), S61.feat_matrix(ev, h2["score_model"]["features"]))
    a4 = S64.frame_rows(ev, dec[dec.row == "4a"].set_index("pair_id").rot)
    meta = pairs.loc[a4.pair_id]
    base = pd.DataFrame({"pair_id": a4.pair_id.to_numpy(), "is_pos": a4.is_pos.to_numpy(), "score": a4.S.to_numpy(),
                         "err_m": np.where(a4.is_pos, a4.err_m, np.nan), "calls": 1,
                         "block": a4.block.to_numpy(), "terrain": meta.terrain.to_numpy(),
                         "agl_m": meta.agl_m.to_numpy(), "gt_heading": meta.gt_heading.to_numpy(),
                         "region": a4.region.to_numpy()})
    out.append(base.assign(row="5", accept=base.score >= h2v["rows"]["5a"]["tau"]))
    sig = np.maximum(a4.spread_m.fillna(50.0).to_numpy(float), h3p["sigma_floor_m"])
    out.append(base.assign(row="6", accept=base.score > h3p["tau"], radius_m=h3p["q"][0] * sig))
    f = filt.copy()
    f["pair_id"] = f.region + "_" + f.num.map("{:04d}".format) + "_pos"
    fm = pairs.loc[f.pair_id]
    out.append(pd.DataFrame({"row": "7", "pair_id": f.pair_id.to_numpy(), "is_pos": True, "accept": True,
                             "score": -f.spread.fillna(f.radius6).to_numpy(), "err_m": f.err7.to_numpy(),
                             "calls": f.matched.astype(int).to_numpy() if "matched" in f else 1,
                             "block": (f.region + "_" + (f.num // 50).astype(str)).to_numpy(),
                             "terrain": fm.terrain.to_numpy(), "agl_m": fm.agl_m.to_numpy(),
                             "gt_heading": fm.gt_heading.to_numpy(), "region": f.region.to_numpy(),
                             "radius_m": (f.kappa * f.spread).to_numpy() if "kappa" in f else np.nan}))
    return pd.concat(out, ignore_index=True)


def row_metrics(g: pd.DataFrame, B: int) -> dict:
    pos = g[g.is_pos]
    acc = pos.accept.to_numpy(bool)
    e = pos.err_m.to_numpy(float)
    d = {f"succ{k}": float((acc & (e <= k)).mean()) for k in (10, 25, 50)}
    d["med_err"] = float(np.nanmedian(np.where(np.isfinite(e), e, np.inf)))
    d["n_pos"], d["n_neg"] = len(pos), int((~g.is_pos).sum())
    good = g.is_pos.to_numpy() & (g.err_m.fillna(np.inf).to_numpy() <= 25)
    d["auroc"] = M.auroc(g.score.to_numpy(float), good) if (~g.is_pos).any() else np.nan
    d["far_neg"] = float(g[~g.is_pos].accept.mean()) if (~g.is_pos).any() else np.nan
    d["calls"] = float(g.calls.mean())
    succ = acc & (e <= 25)
    _, lo, hi = ST.cluster_bootstrap_ci(lambda i: succ[i].mean(), pos.block.to_numpy(), B=B, seed=0)
    d["succ25_ci"] = [lo, hi]
    if "radius_m" in g and g.radius_m.notna().any():
        ap = pos[pos.accept & pos.radius_m.notna()]
        d["coverage"] = float((ap.err_m <= ap.radius_m).mean())
        d["radius_med"] = float(ap.radius_m.median())
    return d


def compare(L: pd.DataFrame) -> list[dict]:
    """Ardışık satırlar: yalnız iki satırda da bulunan pozitif çiftler (7 yalnız pozitif kareler)."""
    pos = L[L.is_pos]
    ok = pos.assign(ok=pos.accept & (pos.err_m <= 25)).pivot(index="pair_id", columns="row", values="ok")
    er = pos.assign(e=pos.err_m.fillna(1e6)).pivot(index="pair_id", columns="row", values="e")
    res = []
    for a, b in PAIRS_CMP:
        m = ok[[a, b]].dropna().index
        mc = ST.mcnemar(ok.loc[m, a].astype(bool), ok.loc[m, b].astype(bool))
        w = st.wilcoxon(er.loc[m, a], er.loc[m, b]) if (er.loc[m, a] != er.loc[m, b]).any() else None
        res.append({"a": a, "b": b, "n": len(m), "a_only": mc["a_only"], "b_only": mc["b_only"], "p_mcnemar": mc["p"],
                    "med_a": float(er.loc[m, a].median()), "med_b": float(er.loc[m, b].median()),
                    "p_wilcoxon": float(w.pvalue) if w else 1.0})
    ph = ST.holm([r["p_mcnemar"] for r in res] + [r["p_wilcoxon"] for r in res])
    for i, r in enumerate(res):
        r["p_mcnemar_holm"], r["p_wilcoxon_holm"] = float(ph[i]), float(ph[len(res) + i])
    return res


def breakdowns(L: pd.DataFrame, agl_edges: list[float]) -> dict:
    pos = L[L.is_pos].copy()
    pos["irtifa"] = pd.cut(pos.agl_m, [-np.inf, *agl_edges, np.inf],
                           labels=[f"< {agl_edges[0]:.0f} m", f"{agl_edges[0]:.0f}–{agl_edges[1]:.0f} m",
                                   f"> {agl_edges[1]:.0f} m"]).astype(str)
    card = np.array(["kuzey (315–45°)", "doğu (45–135°)", "güney (135–225°)", "batı (225–315°)"])
    pos["yön"] = card[(((pos.gt_heading % 360) + 45) // 90 % 4).astype(int)]
    out = {}
    for col in ("terrain", "irtifa", "yön", "region"):
        tab = {}
        for (k, r), g in pos.groupby([col, "row"]):
            e = g.err_m.to_numpy(float)
            tab.setdefault(str(k), {})[r] = {"n": len(g), "succ25": float((g.accept.to_numpy(bool) & (e <= 25)).mean()),
                                             "med_err": float(np.nanmedian(np.where(np.isfinite(e), e, np.inf)))}
        out[col] = tab
    return out


def report(split: str, L: pd.DataFrame, hyp: dict, outdir: Path, B: int, speed: dict | None) -> None:
    pairs = pd.read_csv("data/pairs/pairs_v1.csv", dtype={"region": str})
    ref = pairs[pairs.split.isin(["train", "calib"]) & (pairs.label == "pos")].agl_m
    edges = [float(ref.quantile(1 / 3)), float(ref.quantile(2 / 3))]
    rm = {r: row_metrics(g, B) for r, g in L.groupby("row")}
    cmp_ = compare(L)
    br = breakdowns(L, edges)
    title = "TEST (02, 03, 06), tek sefer" if split == "test" else "PROVA (calib 05, 11) — kod doğrulaması"
    T = [f"# Faz 6 — yedi satırlık ablasyon, {title}", "",
         ("Başarı@k = kabul ∧ hata ≤ k (pozitifler); medyan hata tüm pozitiflerde kabulden bağımsız (satır 7: filtre "
         "kestirimi, yalnız pozitif kareler). AUROC: skor ~ (pozitif ∧ hata ≤ 25 m). GA: 50 karelik blok bootstrap."), "",
         ("| # | yöntem | başarı@10 | başarı@25 [GA] | başarı@50 | medyan hata | AUROC | FAR-neg | çağrı/kare | "
         "kapsama | yarıçap medyan | ms/kare |"), "|---|" + "---|" * 11]
    for r, nm in ROWS.items():
        if r not in rm:
            continue
        d = rm[r]
        ms = speed.get("rows", {}).get(r) if speed else None
        T.append(f"| {r} | {nm} | {d['succ10']:.3f} | {d['succ25']:.3f} [{d['succ25_ci'][0]:.3f}, {d['succ25_ci'][1]:.3f}] | "
                 f"{d['succ50']:.3f} | {d['med_err']:.1f} m | "
                 + (f"{d['auroc']:.3f}" if np.isfinite(d["auroc"]) else "–") + " | "
                 + (f"{d['far_neg']:.3f}" if np.isfinite(d["far_neg"]) else "–") + f" | {d['calls']:.2f} | "
                 + (f"{d['coverage']:.3f}" if "coverage" in d else "–") + " | "
                 + (f"{d['radius_med']:.0f} m" if "radius_med" in d else "–") + " | "
                 + (f"{ms:.0f}" if ms else "–") + " |")
    T += ["", "Ardışık satırlar (aynı pozitif çiftler), Holm düzeltmeli:", "",
          "| karşılaştırma | n | yalnız ilki doğru | yalnız ikincisi | McNemar p (Holm) | medyan hata | Wilcoxon p (Holm) |",
          "|---|---|---|---|---|---|---|"]
    for c in cmp_:
        T.append(f"| {c['a']} vs {c['b']} | {c['n']} | {c['a_only']} | {c['b_only']} | {c['p_mcnemar_holm']:.2e} | "
                 f"{c['med_a']:.1f} vs {c['med_b']:.1f} m | {c['p_wilcoxon_holm']:.2e} |")
    for col, nm, hd in (("terrain", "Arazi", "arazi"), ("irtifa", "İrtifa (AGL)", "irtifa"),
                        ("yön", "Yön (görüntü üstünün pusula yönü)", "yön"), ("region", "Bölge", "bölge")):
        T += ["", f"{nm} kırılımı — başarı@25 · medyan hata (m):", "",
              "| " + hd + " | n poz. | " + " | ".join(f"satır {r}" for r in ROWS) + " |", "|---|---|" + "---|" * len(ROWS)]
        for k, rows in br[col].items():
            n = rows.get("2", rows.get("4", {})).get("n", 0)
            T.append(f"| {k} | {n} | " + " | ".join(
                f"{rows[r]['succ25']:.3f} · {rows[r]['med_err']:.0f}" if r in rows else "–" for r in ROWS) + " |")
    T += ["", "## Hipotezler", ""]
    for h, v in hyp.items():
        T.append(f"- **{h}: {'sağlandı ✓' if v['ok'] else 'sağlanmadı ✗'}** — {v['text']}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"ablation_{split}.md").write_text("\n".join(T) + "\n", encoding="utf-8")
    (outdir / f"ablation_{split}.json").write_text(json.dumps(
        {"rows": rm, "compare": cmp_, "breakdowns": br, "agl_edges": edges, "hypotheses": hyp},
        indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    L.to_csv(outdir / f"ablation_long_{split}.csv", index=False)
    print("\n".join(T))


def hypotheses(h1j: dict, h2j: dict, h3j: dict, h4j: dict) -> dict:
    h1, h2, h3, h4 = h1j["h1"], h2j["h2"], h3j["h3"], h4j["h4"]
    return {
        "H1": {"ok": bool(h1["H1"]), "text": f"(i) medyan yön hatası {h1['heading_median_deg']:.1f}° {'✓' if h1['i'] else '✗'}; "
               f"(ii) çağrı/pozitif kare {h1['calls_per_pos_frame']:.2f} {'✓' if h1['ii'] else '✗'}; (iii) başarı@25 farkı "
               f"{h1['diff_vs_ref']:+.3f} [{h1['diff_ci'][0]:+.3f}, {h1['diff_ci'][1]:+.3f}] {'✓' if h1['iii'] else '✗'}"},
        "H2": {"ok": bool(h2["H2"]), "text": f"(i) yanlış kabul McNemar p = {h2['far_p']:.2e}, 5a < 4a: {'evet' if h2['far_5a_lower'] else 'hayır'} "
               f"{'✓' if h2['i'] else '✗'}; (ii) AUROC farkı {h2['dauc'][0]:+.3f} [{h2['dauc'][1]:+.3f}, {h2['dauc'][2]:+.3f}] "
               f"{'✓' if h2['ii'] else '✗'}"},
        "H3": {"ok": bool(h3["H3"]), "text": f"(i) FAR-neg {h3['far_neg']:.3f} {'✓' if h3['i'] else '✗'}; (ii) kapsama "
               f"{h3['coverage']:.3f} {'✓' if h3['ii'] else '✗'}; (iii) zor çeyrek {h3['coverage_hard']:.3f} "
               f"{'✓' if h3['iii'] else '✗'}"},
        "H4": {"ok": bool(h4["H4"]), "text": f"(i) medyan hata azalması {h4['reduction']:.3f}, Wilcoxon p = "
               f"{h4['wilcoxon_p']:.1e} {'✓' if h4['i'] else '✗'}; (ii) kapsama {h4['coverage']:.3f} {'✓' if h4['ii'] else '✗'}"},
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rehearse", action="store_true")
    g.add_argument("--final", action="store_true")
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    h3 = yaml.safe_load(Path("configs/h3.yaml").read_text(encoding="utf-8"))
    model = h3["system"]["m1_model"]
    py = sys.executable
    if a.final:
        from guvenloc import final_guard
        final_guard.check("90_ablation")
        for cmd in (["scripts/56_m1c_roma.py", "--model", model, "--split", "test"],
                    ["scripts/61_m2_score.py", "--split", "test"],
                    ["scripts/64_m3_conformal.py", "--final"], ["scripts/72_m4_filter.py", "--final"]):
            print(f"\n=== {' '.join(cmd)}", flush=True)
            subprocess.run([py, *cmd], check=True)
        split, src = "test", Path("results/faz6/final")
        h1j = json.loads((src / f"roma_row4_{model}_test.json").read_text(encoding="utf-8"))
        h2j = json.loads((src / "h2_test.json").read_text(encoding="utf-8"))
        h3j = json.loads((src / "h3_test.json").read_text(encoding="utf-8"))
        h4j = json.loads((src / "h4_test.json").read_text(encoding="utf-8"))
        filt = pd.read_csv(src / "h4_test_7.csv", dtype={"region": str})
        out = src
    else:
        split, src = "calib", Path("results/faz2/m1")
        reh = Path("results/faz6/rehearsal")
        h1j = json.loads((src / f"roma_row4_{model}_calib.json").read_text(encoding="utf-8"))
        h2j = json.loads(Path("results/faz3/m2/h2_calib.json").read_text(encoding="utf-8"))
        h3j = json.loads((reh / "h3_test.json").read_text(encoding="utf-8"))
        h4j = json.loads((reh / "h4_test.json").read_text(encoding="utf-8"))
        filt = pd.read_csv(reh / "h4_test_7.csv", dtype={"region": str})
        out = reh
    h3p = json.loads(Path("results/faz4/m3/m3_params_h3_v1.json").read_text(encoding="utf-8"))
    filt["kappa"] = h4j["kappa"]
    L = build(split, src, h2j, h3p, filt, model)
    speed_p = Path("results/faz6/speed.json")
    speed = json.loads(speed_p.read_text(encoding="utf-8")) if speed_p.exists() else None
    report(split, L, hypotheses(h1j, h2j, h3j, h4j), out, a.B, speed)
    if a.final:
        final_guard.done("90_ablation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
