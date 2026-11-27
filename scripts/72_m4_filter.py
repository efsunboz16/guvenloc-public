"""Faz 5B–5D – parçacık filtresi uçuşlarda: eğitimde hiper-parametre seçimi, calib kuru denemesi (configs/h4.yaml).

  python scripts/72_m4_filter.py --tune     # eğitim bölgeleri (01, 04, 08, 09, 10): ızgara → results/faz5/m4/tune.json
  python scripts/72_m4_filter.py --dry      # seçilen ayarla calib (05, 11): satır 6 / 7 / 7-noupd / 7-cv, κ
  python scripts/72_m4_filter.py --gamma    # 5C: γ eşikli uydu eşleştirmesi, calib (F5-6) → gamma.md / .json
  python scripts/72_m4_filter.py --final    # Faz 6: test (02, 03, 06), tek sefer
  python scripts/72_m4_filter.py --traj     # seçilen ayarla eğitim + calib yörüngeleri → traj.csv (izleme paneli)
  (test bölgeleri 02, 03, 06: Faz 6, tek sefer — bu betik reddeder)

Girdi: 71_m4_measure.py (kare başına P(x, y), tek kare kararı), 70_m4_sequences.py (drone↔drone hareket).
Hata: GT = bölge CSV'si (drone_lat/lon), yerel ENU. Satır 6 = tek kare hacim tepesi (kabulden bağımsız).
Satır 7 = filtre; ilk kabul edilen kareden önce satır 6. Seçim ölçütü (eğitim): havuzlanmış medyan hata.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc import m4_filter as F
from guvenloc import stats as ST
from guvenloc.metrics import log_run

OUT = Path("results/faz5/m4")
TEST = {"02", "03", "06"}
MIN_INL = 50
GRID = {"lik_power": [0.5, 1.0, 2.0], "sigma_pos_rate": [0.1, 0.2, 0.3], "eps": [1e-4, 1e-3, 1e-2],
        "theta_meas_sigma_deg": [5.0, 1e9], "reset_lik": [1e-4, 1e-3, 1e-2]}
ODO_WIN, ODO_LO, ODO_HI = 5, 0.5, 2.0       # F5-3: odometri kapısı (adım uzunluğu son 5 iyi adımın medyanına göre)


def load_region(rid: str, params: dict, final: bool = False) -> dict:
    if rid in TEST and not final:
        sys.exit("test bölgeleri Faz 6'da (h4.yaml)")
    d = pd.read_csv(OUT / f"meas_{rid}.csv", dtype={"region": str}).sort_values("num").reset_index(drop=True)
    P = np.load(OUT / f"meas_{rid}.npz")["P"]
    seq = pd.read_csv(Path("results/faz6/prep/seq_motion_test.csv") if rid in TEST else OUT / "seq_stats.csv",
                      dtype={"region": str})
    seq = seq[seq.region == rid].set_index("t1")
    lat0, lon0 = float(d.tile_lat[0]), float(d.tile_lon[0])
    ge, gn = F.enu(lat0, lon0, d.drone_lat, d.drone_lon)
    ce, cn = F.enu(lat0, lon0, d.tile_lat, d.tile_lon)
    meas, single = [], []
    for k, r in enumerate(d.itertuples()):
        n = int(np.ceil(r.tile_px / (r.cell_m / (r.tile_size_m / r.tile_px))))
        m = F.GridMeas(P[k, :n, :n], (float(ce[k]), float(cn[k])), int(r.tile_px), r.tile_size_m / r.tile_px,
                       float(r.cell_m))
        meas.append(m)
        single.append(m.to_en(r.x, r.y) if np.isfinite(r.x) else (np.nan, np.nan))
    mot, good = [], []
    for k, r in enumerate(d.itertuples()):
        s = seq.loc[r.num] if k and r.num in seq.index else None
        ok = s is not None and s.n_inliers >= MIN_INL and np.isfinite(s.fwd_px) and int(s.t0) == int(d.num[k - 1])
        mpp = float(d.drone_m_per_px[k - 1]) if k else np.nan
        mv = (s.fwd_px * mpp, s.right_px * mpp, s.dtheta) if ok else None
        if mv is not None and len(good) >= ODO_WIN:      # F5-3: takılı/sıçrayan adım → son iyi adım (sabit hız)
            med = float(np.median([np.hypot(g[0], g[1]) for g in good[-ODO_WIN:]]))
            if not ODO_LO * med <= np.hypot(mv[0], mv[1]) <= ODO_HI * med:
                mv = (good[-1][0], good[-1][1], 0.0)
            else:
                good.append(mv)
        elif mv is not None:
            good.append(mv)
        elif good:
            mv = (good[-1][0], good[-1][1], 0.0)
        mot.append(mv)
    return {"d": d, "meas": meas, "single": np.array(single), "gt": np.column_stack([ge, gn]), "motion": mot,
            "q": float(params["q"][0])}


def run(R: dict, cfg: F.FilterCfg, mode: str = "7", gamma: float = 0.0, snaps: dict | None = None) -> pd.DataFrame:
    """mode: 7 (plan), 7-noupd (M3 reddinde de güncelle), 7-cv (sabit hız; drone↔drone eşleşmesi kullanılmaz).
    gamma (5C, F5-6): tahminden sonra yayılım (ağırlıklı RMS, m) ≤ γ ise uydu eşleştirmesi yapılmaz (matched = False)."""
    d, pf = R["d"], F.ParticleFilter(cfg)
    rows, prev = [], []
    for k, r in enumerate(d.itertuples()):
        acc, reset, matched = bool(r.accept), False, True
        if pf.started:
            mv = R["motion"][k]
            if mode == "7-cv" or mv is None:                # B planı: son iki filtre kestiriminden hız
                if len(prev) >= 2:
                    pf.predict_en(prev[-1][0] - prev[-2][0], prev[-1][1] - prev[-2][1], 0.0)
                else:
                    pf.predict_en(0.0, 0.0, 0.0)
            else:
                pf.predict(*mv)
            if gamma > 0 and pf.estimate()["std_m"] <= gamma:
                matched, acc = False, False                 # uydu eşleştirmesi atlandı: yalnız tahmin
            if acc and np.isfinite(R["single"][k][0]) and pf.explains(R["meas"][k]) < cfg.reset_lik:
                pf.init(*R["single"][k], R["q"] * r.sigma_hat, r.theta)      # sensör sıfırlaması
                reset = True
            elif acc or (mode == "7-noupd" and matched):
                pf.update(R["meas"][k], r.theta if acc else None)
            est = pf.estimate()
            xy, spread, started = (est["e"], est["n"]), est["spread_q90_m"], True
        elif acc and np.isfinite(R["single"][k][0]):
            pf.init(*R["single"][k], R["q"] * r.sigma_hat, r.theta)
            est = pf.estimate()
            xy, spread, started = (est["e"], est["n"]), est["spread_q90_m"], True
        else:
            xy, spread, started = tuple(R["single"][k]), np.nan, False
        prev.append(xy)
        if snaps is not None and r.num in snaps and pf.started:   # 5D: parçacık bulutu anlık görüntüsü
            snaps[r.num] = (pf.x[:, :2].copy(), pf.w.copy())
        g = R["gt"][k]
        rows.append({"region": r.region, "num": r.num, "accept": acc, "started": started,
                     "reset": reset, "matched": matched,
                     "err6": float(np.hypot(*(R["single"][k] - g))), "err7": float(np.hypot(xy[0] - g[0], xy[1] - g[1])),
                     "spread": spread, "radius6": R["q"] * r.sigma_hat,
                     "e6": float(R["single"][k][0]), "n6": float(R["single"][k][1]), "e7": float(xy[0]),
                     "n7": float(xy[1]), "ge": float(g[0]), "gn": float(g[1])})
    return pd.DataFrame(rows)


def tune(params: dict, h4: dict) -> dict:
    regs = {r: load_region(r, params) for r in h4["tuning_regions"]}
    base = F.FilterCfg()
    res = []
    keys = list(GRID)
    for vals in itertools.product(*GRID.values()):
        cfg = replace(base, **dict(zip(keys, vals, strict=True)))
        t0 = time.time()
        df = pd.concat([run(R, cfg) for R in regs.values()])
        res.append({**dict(zip(keys, vals, strict=True)), "med7": float(df.err7.median()),
                    "med6": float(df.err6.median()),
                    "per_region": df.groupby("region").err7.median().round(1).to_dict()})
        print(f"{vals}: medyan 7 {res[-1]['med7']:.1f} m (6: {res[-1]['med6']:.1f}) {time.time() - t0:.0f} s",
              flush=True)
    best = min(res, key=lambda x: x["med7"])
    cfg = replace(base, **{k: best[k] for k in keys})
    out = {"selected": asdict(cfg), "grid": res, "criterion": "eğitim bölgelerinde havuzlanmış medyan hata (satır 7)"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tune.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("seçilen:", asdict(cfg), f"medyan {best['med7']:.1f} m")
    return out


def kappa_for(df: pd.DataFrame, cover: float = 0.9) -> float:
    """err7 ≤ κ · spread oranı ≥ cover olan en küçük κ (filtre başlamış kareler)."""
    s = df[df.started & (df.spread > 0)]
    return float(np.quantile(s.err7 / s.spread, cover, method="higher"))


def summarize(df: pd.DataFrame, kappa: float) -> dict:
    s = df[df.started]
    m6, m7 = float(df.err6.median()), float(df.err7.median())
    w = st.wilcoxon(df.err6, df.err7, alternative="greater")
    blocks = df.region + "_" + (df.num // 50).astype(str)
    e6, e7 = df.err6.to_numpy(), df.err7.to_numpy()
    red = ST.cluster_bootstrap_ci(lambda i: 1 - np.median(e7[i]) / np.median(e6[i]), blocks.to_numpy(), B=2000)
    mc25 = ST.mcnemar(df.err7 <= 25, df.err6 <= 25)
    mc50 = ST.mcnemar(df.err7 <= 50, df.err6 <= 50)
    return {"n": len(df), "n_started": len(s), "med6": m6, "med7": m7, "reduction": 1 - m7 / m6,
            "reduction_ci": [red[1], red[2]], "wilcoxon_p": float(w.pvalue),
            "succ25": [float((df.err6 <= 25).mean()), float((df.err7 <= 25).mean())], "mcnemar25_p": mc25["p"],
            "succ50": [float((df.err6 <= 50).mean()), float((df.err7 <= 50).mean())], "mcnemar50_p": mc50["p"],
            "coverage7": float((s.err7 <= kappa * s.spread).mean()), "kappa": kappa,
            "radius7_median": float((kappa * s.spread).median()),
            "per_region": {r: {"med6": float(g.err6.median()), "med7": float(g.err7.median())}
                           for r, g in df.groupby("region")}}


def dry(params: dict, h4: dict) -> int:
    cfg = F.FilterCfg(**json.loads((OUT / "tune.json").read_text(encoding="utf-8"))["selected"])
    cal = h4["calibration_regions"]
    regs = {r: load_region(r, params) for r in cal}
    crit = h4["criteria"]
    res = {}
    for mode in ("7", "7-noupd", "7-cv"):
        dfs = {r: run(R, cfg, mode) for r, R in regs.items()}
        df = pd.concat(dfs.values(), ignore_index=True)
        df.to_csv(OUT / f"dry_{mode}.csv", index=False)
        k_pool = kappa_for(df)
        res[mode] = {"pooled_in_sample": summarize(df, k_pool)}
        for a, b in (("05", "11"), ("11", "05")):           # κ taşınması: birinde seç, ötekinde kapsama
            res[mode][f"kappa {a} → {b}"] = summarize(dfs[b], kappa_for(dfs[a]))
        print(mode, {k: round(v["reduction"], 3) for k, v in res[mode].items()}, flush=True)
    L = ["# Faz 5 — H4 kuru denemesi (calib 05 + 11; test değil), `configs/h4.yaml`", "",
         f"Filtre ayarı eğitimde seçildi (`tune.json`): {asdict(cfg)}.", "",
         ("Satır 6 = tek kare hacim tepesi (her karede), satır 7 = filtre. Azalma = 1 − med7/med6 [blok bootstrap %95]; "
         "Wilcoxon tek yönlü. Kapsama: filtre başlamış karelerde hata ≤ κ · %90 yayılım. 'havuz' κ'yı aynı veride seçer "
         "(tanım gereği ≈ 0,90); '05 → 11' κ'yı 05'te seçip 11'de ölçer."), "",
         ("| satır | değerlendirme | n | med6 | med7 | azalma [GA] | Wilcoxon p | başarı@25 6/7 | başarı@50 6/7 | κ | "
         "kapsama 7 | yarıçap medyan | H4 (i) (ii) |"), "|---|" + "---|" * 12]
    for mode, blocks in res.items():
        for key, v in blocks.items():
            ok = (v["reduction"] >= crit["median_reduction_min"] and v["wilcoxon_p"] < crit["wilcoxon_p_max"],
                  v["coverage7"] >= crit["coverage_min"])
            L.append(f"| {mode} | {key} | {v['n']} | {v['med6']:.1f} | {v['med7']:.1f} | {v['reduction']:.3f} "
                     f"[{v['reduction_ci'][0]:.3f}, {v['reduction_ci'][1]:.3f}] | {v['wilcoxon_p']:.1e} | "
                     f"{v['succ25'][0]:.3f}/{v['succ25'][1]:.3f} | {v['succ50'][0]:.3f}/{v['succ50'][1]:.3f} | "
                     f"{v['kappa']:.2f} | {v['coverage7']:.3f} | {v['radius7_median']:.0f} m | "
                     + " ".join("✓" if o else "✗" for o in ok) + " |")
    pr = res["7"]["pooled_in_sample"]["per_region"]
    L += ["", "Bölge başına (satır 7): " + "; ".join(f"{r}: {v['med6']:.0f} → {v['med7']:.0f} m" for r, v in pr.items())]
    (OUT / "h4_dry.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "h4_dry.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(L))
    log_run("72_m4_filter.py", Path("configs/h4.yaml"), 0, "05+11", "row7_reduction",
            round(res["7"]["pooled_in_sample"]["reduction"], 5))
    return 0


GAMMAS = [0, 25, 50, 75, 100, 150, 200, 300, 500]


def gamma_sweep(params: dict, h4: dict) -> int:
    """5C (F5-6): γ ızgarası calib'de; κ = F5-5 (h4_dry.json havuz), ayar tune.json. Kural: med ≤ 1,05 × med(γ=0)
    ve kapsama ≥ 0,88 olan en büyük γ."""
    cfg = F.FilterCfg(**json.loads((OUT / "tune.json").read_text(encoding="utf-8"))["selected"])
    kappa = json.loads((OUT / "h4_dry.json").read_text(encoding="utf-8"))["7"]["pooled_in_sample"]["kappa"]
    regs = {r: load_region(r, params) for r in h4["calibration_regions"]}
    res = []
    for gm in GAMMAS:
        df = pd.concat([run(R, cfg, "7", float(gm)) for R in regs.values()], ignore_index=True)
        s = df[df.started]
        res.append({"gamma": gm, "med7": float(df.err7.median()), "med6": float(df.err6.median()),
                    "match_rate": float(df.matched.mean()), "skip_rate": float(1 - df.matched.mean()),
                    "coverage7": float((s.err7 <= kappa * s.spread).mean()),
                    "succ25": float((df.err7 <= 25).mean()), "succ50": float((df.err7 <= 50).mean()),
                    "per_region": {r: {"med7": float(g.err7.median()), "skip": float(1 - g.matched.mean())}
                                   for r, g in df.groupby("region")}})
        print(f"γ = {gm}: medyan {res[-1]['med7']:.1f} m, atlanan {res[-1]['skip_rate']:.3f}, "
              f"kapsama {res[-1]['coverage7']:.3f}", flush=True)
    base = res[0]["med7"]
    ok = [x for x in res if x["med7"] <= 1.05 * base and x["coverage7"] >= h4["criteria"]["coverage_min"]]
    sel = max(ok, key=lambda x: x["gamma"])["gamma"] if ok else 0
    L = ["# Faz 5C — γ eşikli uydu eşleştirmesi (calib 05 + 11), F5-6", "",
         (f"Filtre ayarı `tune.json`, κ = {kappa:.2f} (F5-5). Tahminden sonra yayılım ≤ γ → uydu eşleştirmesi yok. "
         "Kural: medyan ≤ 1,05 × medyan(γ = 0) ve kapsama ≥ 0,88 olan en büyük γ."), "",
         "| γ (m) | uydu eşleştirmesi | atlanan | medyan hata | başarı@25 | başarı@50 | kapsama | 05 / 11 medyan | kural |",
         "|---|---|---|---|---|---|---|---|---|"]
    for x in res:
        ok_x = x["med7"] <= 1.05 * base and x["coverage7"] >= h4["criteria"]["coverage_min"]
        pr = x["per_region"]
        L.append(f"| {x['gamma']} | {x['match_rate']:.3f} | {x['skip_rate']:.3f} | {x['med7']:.1f} m | {x['succ25']:.3f} | "
                 f"{x['succ50']:.3f} | {x['coverage7']:.3f} | {pr['05']['med7']:.0f} / {pr['11']['med7']:.0f} m | "
                 f"{'✓' if ok_x else '✗'}{' **seçilen**' if x['gamma'] == sel else ''} |")
    L += ["", f"Tek kare (satır 6) medyanı {res[0]['med6']:.1f} m. **Seçilen γ = {sel} m.**"]
    (OUT / "gamma.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "gamma.json").write_text(json.dumps({"selected": sel, "kappa": kappa, "grid": res}, indent=2,
                                               ensure_ascii=False), encoding="utf-8")
    print("\n".join(L))
    return 0


def final(params: dict, h4: dict, rehearse: bool = False) -> int:
    """H4 TEST (02, 03, 06): tune.json ayarı, tohum 0, κ (F5-5) ve γ (F5-7) sabit; yeniden ayar yok.
    rehearse: aynı kod calib bölgelerinde, koruyucusuz, results/faz6/rehearsal/ (kod doğrulaması)."""
    from guvenloc import final_guard
    if not rehearse:
        final_guard.check("72_h4")
    cfg = F.FilterCfg(**json.loads((OUT / "tune.json").read_text(encoding="utf-8"))["selected"])
    kappa = json.loads((OUT / "h4_dry.json").read_text(encoding="utf-8"))["7"]["pooled_in_sample"]["kappa"]
    gamma = float(json.loads((OUT / "gamma.json").read_text(encoding="utf-8"))["selected"])
    regs = {r: load_region(r, params, final=True)
            for r in (h4["calibration_regions"] if rehearse else sorted(TEST))}
    fo = Path("results/faz6/rehearsal" if rehearse else "results/faz6/final")
    fo.mkdir(parents=True, exist_ok=True)
    crit = h4["criteria"]
    res = {}
    for mode, gm in (("7", 0.0), ("7-noupd", 0.0), ("7-cv", 0.0), ("7-γ", gamma)):
        dfs = {r: run(R, cfg, "7" if mode == "7-γ" else mode, gm) for r, R in regs.items()}
        df = pd.concat(dfs.values(), ignore_index=True)
        df.to_csv(fo / f"h4_test_{mode}.csv", index=False)
        res[mode] = {"pooled": summarize(df, kappa) | {"skip_rate": float(1 - df.matched.mean())}}
        for r, g in dfs.items():
            res[mode][r] = summarize(g, kappa) | {"skip_rate": float(1 - g.matched.mean())}
    v = res["7"]["pooled"]
    h4res = {"reduction": v["reduction"], "wilcoxon_p": v["wilcoxon_p"], "coverage": v["coverage7"],
             "i": v["reduction"] >= crit["median_reduction_min"] and v["wilcoxon_p"] < crit["wilcoxon_p_max"],
             "ii": v["coverage7"] >= crit["coverage_min"]}
    h4res["H4"] = bool(h4res["i"] and h4res["ii"])
    L = [f"# Faz 6 — H4 {'PROVA (calib)' if rehearse else 'TEST (02, 03, 06)'}, `configs/h4.yaml`", "",
         f"Filtre ayarı `tune.json` (tohum 0), κ = {kappa:.2f} (F5-5), γ = {gamma:.0f} m (F5-7, satır 7-γ). Tek sefer.", "",
         ("| satır | değerlendirme | n | med6 | med7 | azalma [GA] | Wilcoxon p | başarı@25 6/7 | başarı@50 6/7 | "
         "kapsama 7 | yarıçap medyan | atlanan | H4 (i) (ii) |"), "|---|" + "---|" * 12]
    for mode, blocks in res.items():
        for key, x in blocks.items():
            ok = (x["reduction"] >= crit["median_reduction_min"] and x["wilcoxon_p"] < crit["wilcoxon_p_max"],
                  x["coverage7"] >= crit["coverage_min"])
            L.append(f"| {mode} | {key} | {x['n']} | {x['med6']:.1f} | {x['med7']:.1f} | {x['reduction']:.3f} "
                     f"[{x['reduction_ci'][0]:.3f}, {x['reduction_ci'][1]:.3f}] | {x['wilcoxon_p']:.1e} | "
                     f"{x['succ25'][0]:.3f}/{x['succ25'][1]:.3f} | {x['succ50'][0]:.3f}/{x['succ50'][1]:.3f} | "
                     f"{x['coverage7']:.3f} | {x['radius7_median']:.0f} m | {x['skip_rate']:.3f} | "
                     + " ".join("✓" if o else "✗" for o in ok) + " |")
    L += ["", (f"**H4 {'sağlandı' if h4res['H4'] else 'sağlanmadı'}** (satır 7, havuz): (i) azalma "
          f"{h4res['reduction']:.3f} ≥ {crit['median_reduction_min']} ve p = {h4res['wilcoxon_p']:.1e} < "
          f"{crit['wilcoxon_p_max']} {'✓' if h4res['i'] else '✗'}; (ii) kapsama {h4res['coverage']:.3f} ≥ "
          f"{crit['coverage_min']} {'✓' if h4res['ii'] else '✗'}.")]
    (fo / "h4_test.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (fo / "h4_test.json").write_text(json.dumps({"results": res, "h4": h4res, "kappa": kappa, "gamma": gamma},
                                                indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("\n".join(L))
    if not rehearse:
        log_run("72_m4_filter.py", Path("configs/h4.yaml"), 0, "test", "row7_reduction", round(h4res["reduction"], 5))
        final_guard.done("72_h4")
    return 0


def traj(params: dict, h4: dict) -> None:
    cfg = F.FilterCfg(**json.loads((OUT / "tune.json").read_text(encoding="utf-8"))["selected"])
    regs = h4["tuning_regions"] + h4["calibration_regions"]
    df = pd.concat([run(load_region(r, params), cfg) for r in regs], ignore_index=True)
    df.to_csv(OUT / "traj.csv", index=False)
    print(f"yazıldı: {OUT / 'traj.csv'} ({len(df)} kare)")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--traj", action="store_true")
    ap.add_argument("--gamma", action="store_true")
    ap.add_argument("--final", action="store_true", help="Faz 6: test bölgeleri, tek sefer (final_guard)")
    ap.add_argument("--rehearse", action="store_true", help="--final kodunu calib'de dene (koruyucusuz)")
    a = ap.parse_args()
    h4 = yaml.safe_load(Path("configs/h4.yaml").read_text(encoding="utf-8"))
    params = json.loads(Path("results/faz4/m3/m3_params_h3_v1.json").read_text(encoding="utf-8"))
    if a.tune:
        tune(params, h4)
    if a.final or a.rehearse:
        return final(params, h4, rehearse=a.rehearse)
    if a.gamma:
        return gamma_sweep(params, h4)
    if a.traj:
        traj(params, h4)
    if a.dry:
        return dry(params, h4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
