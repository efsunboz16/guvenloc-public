"""Faz 4G – uçtan uca hat: parametre dışa aktarımı ve 100 karelik uçtan uca test (`guvenloc/pipeline.py`).

  python scripts/67_m3_pipeline.py --export        # S (h2_v3, eğitim) + τ, q̂ (h3_v1 satır 6, 05+11) → JSON
  python scripts/67_m3_pipeline.py --e2e 100       # kalibrasyon bölgelerinden 100 kare, canlı M1-C + RoMa

Uçtan uca test bir DEĞERLENDİRME değildir (τ ve q̂ aynı calib verisinden): canlı hattın önbellekli çevrimdışı
hatla (30 → 40 → 60 → 61 → 64) aynı kararı verdiğini doğrular. Karşılaştırılanlar: M1-C dönmesi (4a kararı),
n_inliers, S, kabul, hacim tepesi (m), σ̂; ayrıca aşama başına süre.
Test bölgeleri (02, 03, 06) kullanılmaz.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from guvenloc import m3_conformal as C
from guvenloc import pipeline as P
from guvenloc.io_visloc import crop_square_m, load_region
from guvenloc.m1_canon import prep_square
from guvenloc.metrics import log_run


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S61 = _load("m2s61", "scripts/61_m2_score.py")
S64 = _load("m3s64", "scripts/64_m3_conformal.py")
OUT = Path("results/faz4/m3")
PARAMS = OUT / "m3_params_h3_v1.json"
DATA = Path("data/UAV_VisLoc_dataset")
TEST = {"02", "03", "06"}


def calib_frame(h2, h3, pairs):
    """Çevrimdışı satır 6 verisi: calib (çift, 4a dönmesi) + S; ayrıca eğitilmiş S modeli ve eğitim entropi dolgusu."""
    tr, ca = S61.load_split("train", pairs), S61.load_split("calib", pairs)
    model = S64.fit_S(tr, h2)
    ca["S"] = S61.predict(model, S61.feat_matrix(ca, h2["score_model"]["features"]))
    dec = pd.read_csv(S61.M1_DECISIONS.format(model=h3["system"]["m1_model"]))
    ca_f = S64.frame_rows(ca, dec[dec.row == "4a"].set_index("pair_id").rot)
    return model, float(tr.entropy_bits.max()), ca_f


def export(h2, h3, pairs) -> dict:
    model, ent_fill, ca_f = calib_frame(h2, h3, pairs)
    d = ca_f[ca_f.region.isin(h3["calibration_regions"])]
    s, err, sig, _, _ = S64.variants(d, float(h3["system"]["sigma_floor_m"]))["6"]
    pos = d.is_pos.to_numpy()
    cal = C.fit(s[~pos], s[pos], err[pos], sig[pos], float(h3["alpha"]), float(h3["beta"]))
    base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    mcfg = yaml.safe_load(Path("configs/m2.yaml").read_text(encoding="utf-8"))
    params = {"version": h3["version"], "h2_version": h2["version"], "m1_model": h3["system"]["m1_model"],
              "alpha": h3["alpha"], "beta": h3["beta"], "calibration_regions": h3["calibration_regions"],
              "score_model": {"features": h2["score_model"]["features"], "w": model["w"].tolist(),
                              "mu": model["mu"].tolist(), "sd": model["sd"].tolist()},
              "entropy_fill": ent_fill, "tau": cal.tau, "q": cal.q.tolist(), "edges": cal.edges.tolist(),
              "n_neg": cal.n_neg, "n_pos_acc": cal.n_pos_acc.tolist(),
              "sigma_floor_m": float(h3["system"]["sigma_floor_m"]), "ransac_px": 5.0,
              "volume_cfg": mcfg | {"sigma_scale": float(base["romav2"]["sigma_scale"])},
              "romav2": {"setting": base["romav2"]["setting_cache"], "n_matches": int(base["romav2"]["n_matches"]),
                         "sigma_scale_matcher": 1.0}}
    loc = P.Localizer(params)                                # RoMa'sız denetim: öznitelik satırı → karar
    dec = [loc.decide(f, int(f["rot"]), 1.0) for f in d.to_dict("records")]
    dS = np.abs(np.array([x.score for x in dec]) - s)
    fin = np.isfinite(d.entropy_bits.to_numpy())
    params["check"] = {"max_abs_dS": float(dS[fin].max()), "n_entropy_nan": int((~fin).sum()),
                       "accept_agree": float(np.mean([x.accept for x in dec] == (s > cal.tau))),
                       "radius_agree": float(np.mean(np.isclose([x.radius_m for x in dec], cal.radius(sig))))}
    print("denetim (çevrimdışı satırlar → Localizer.decide):", params["check"])
    OUT.mkdir(parents=True, exist_ok=True)
    PARAMS.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"yazıldı: {PARAMS}  τ = {cal.tau:.4f}, q̂ = {cal.q.tolist()}, n⁻ = {cal.n_neg}, "
          f"n⁺ kabul = {cal.n_pos_acc.tolist()}")
    return params


class M1Live:
    """M1-C birleşimi (+TTA) tek kare × tek karo için: 51_m1_prep (JPEG 1/4 indirgeme + merkez kare 321) ve
    56_m1c_roma.tiles_for (513 px karo) ile aynı ön işleme; olasılıklar 56.probs_arrays'den."""

    def __init__(self, model: str):
        self.R56 = _load("m1c56", "scripts/56_m1c_roma.py")
        self.R56.C55.MODE["veri"] = "gercek"
        tta = model.endswith("@tta")
        self.angles = self.R56.TTA_DEG if tta else (0.0,)
        self.nets = [self.R56._net(n) for n in model.removesuffix("@tta").split("+")]

    def __call__(self, frame_bgr_r4: np.ndarray, tile513: np.ndarray) -> np.ndarray:
        d = torch.from_numpy(prep_square(frame_bgr_r4, 321)[None]).cuda()
        s = torch.from_numpy(np.ascontiguousarray(tile513[None, ..., :3])).cuda()
        return self.R56.probs_arrays(self.nets, self.angles, d, s)[0][0]


def e2e(h2, h3, pairs_all, n_frames: int) -> int:
    from guvenloc.matcher import Matcher

    pairs = pairs_all.set_index("pair_id")
    params = json.loads(PARAMS.read_text(encoding="utf-8"))
    _, _, ca_f = calib_frame(h2, h3, pairs)
    off = ca_f.set_index("pair_id")
    cp = pairs_all[(pairs_all.split == "calib") & pairs_all.region.isin(h3["calibration_regions"])]
    assert not TEST & set(cp.region) and not TEST & set(cp.tile_region)
    frames = []
    for _r, g in cp.groupby("region"):                      # bölge başına eşit aralıklı kareler
        dp = sorted(g.drone_path.unique())
        frames += [dp[i] for i in np.linspace(0, len(dp) - 1, n_frames // 2).round().astype(int)]
    rc = params["romav2"]
    loc = P.Localizer(params, Matcher(setting=rc["setting"], sigma_scale=rc["sigma_scale_matcher"], seed=0),
                      rc["n_matches"])
    m1 = M1Live(params["m1_model"])
    regions = {}
    rows, t_all = [], time.time()
    for k, dp in enumerate(frames):
        t0 = time.perf_counter()
        bgr = cv2.imread(str(DATA / dp))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bgr4 = cv2.imread(str(DATA / dp), cv2.IMREAD_REDUCED_COLOR_4)
        t_io = time.perf_counter() - t0
        for r in cp[cp.drone_path == dp].itertuples():
            if r.tile_region not in regions:
                regions[r.tile_region] = load_region(DATA, r.tile_region)
            reg = regions[r.tile_region]
            t1 = time.perf_counter()
            tile = np.ascontiguousarray(crop_square_m(reg, r.tile_lat, r.tile_lon, r.tile_size_m, int(r.tile_px)))
            tile513 = crop_square_m(reg, r.tile_lat, r.tile_lon, r.tile_size_m, 513)
            t2 = time.perf_counter()
            probs = m1(bgr4, tile513)
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            dcs = loc.localize(rgb, tile, float(r.scale_s), float(r.tile_size_m), rot_probs=probs,
                               geo={"tile_lat": r.tile_lat, "tile_lon": r.tile_lon})
            t4 = time.perf_counter()
            o = off.loc[r.pair_id]
            mpp = r.tile_size_m / r.tile_px
            o_rot = int(o.rot)
            if dcs.rot != o_rot:                            # dönme farklıysa karşılaştırma için de çevrimdışı dönmede
                dcs_o = loc.localize(rgb, tile, float(r.scale_s), float(r.tile_size_m), rot=o_rot)
            else:
                dcs_o = dcs
            err = (float(np.hypot(dcs.xy[0] - r.gt_x, dcs.xy[1] - r.gt_y) * mpp) if r.label == "pos" else np.nan)
            ll_err = np.nan
            if r.label == "pos" and dcs.latlon is not None:
                dn = np.radians(dcs.latlon[0] - r.drone_lat) * P.EARTH_R
                de = np.radians(dcs.latlon[1] - r.drone_lon) * P.EARTH_R * np.cos(np.radians(r.drone_lat))
                ll_err = float(np.hypot(dn, de))
            rows.append({"pair_id": r.pair_id, "region": r.region, "label": r.label, "rot": dcs.rot, "rot_off": o_rot,
                         "S": dcs.score, "S_off": float(o.S), "S_same_rot": dcs_o.score,
                         "accept": dcs.accept, "accept_off": bool(o.S > params["tau"]),
                         "accept_same_rot": dcs_o.accept,
                         "n_inl": dcs_o.n_inliers, "n_inl_off": int(o.n_inliers),
                         "peak_shift_m": float(np.hypot(dcs_o.xy[0] - o.x, dcs_o.xy[1] - o.y) * mpp),
                         "sigma": dcs_o.sigma_hat, "sigma_off": max(50.0 if np.isnan(o.spread_m) else o.spread_m,
                                                                    params["sigma_floor_m"]),
                         "err_m": err, "err_latlon_m": ll_err, "radius_m": dcs.radius_m,
                         "t_io": t_io, "t_tile": t2 - t1, "t_m1": t3 - t2, "t_loc": t4 - t3})
        if (k + 1) % 10 == 0:
            el = time.time() - t_all
            print(f"  kare {k + 1}/{len(frames)} | {el / (k + 1):.1f} s/kare | kalan ≈ "
                  f"{el / (k + 1) * (len(frames) - k - 1) / 60:.1f} dk", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e2e.csv", index=False)
    report(df, params, len(frames))
    return 0


def report(df: pd.DataFrame, params: dict, n_frames: int) -> None:
    pos = df[df.label == "pos"]
    same = df.rot == df.rot_off
    dS = (df.S_same_rot - df.S_off).abs()
    acc_pos = pos[pos.accept]
    stats = {"n_frames": n_frames, "n_pairs": len(df), "rot_agree": float(same.mean()),
             "accept_agree": float((df.accept == df.accept_off).mean()),
             "accept_agree_same_rot": float((df.accept_same_rot == df.accept_off).mean()),
             "dS_median": float(dS.median()), "dS_p95": float(dS.quantile(0.95)),
             "n_inl_rel_diff_median": float(((df.n_inl - df.n_inl_off).abs() / df.n_inl_off.clip(lower=1)).median()),
             "peak_shift_median_m": float(df.peak_shift_m.median()),
             "peak_shift_p95_m": float(df.peak_shift_m.quantile(0.95)),
             "far_neg": float(df[df.label != "pos"].accept.mean()),
             "far_neg_off": float(df[df.label != "pos"].accept_off.mean()),
             "accept_pos": float(pos.accept.mean()),
             "coverage": float((acc_pos.err_m <= acc_pos.radius_m).mean()) if len(acc_pos) else np.nan,
             "err_median_acc_pos_m": float(acc_pos.err_m.median()) if len(acc_pos) else np.nan,
             "latlon_vs_px_err_median_m": float((pos.err_latlon_m - pos.err_m).abs().median()),
             "t_pair_median_s": float((df.t_tile + df.t_m1 + df.t_loc).median()),
             "t_m1_median_s": float(df.t_m1.median()), "t_loc_median_s": float(df.t_loc.median()),
             "t_tile_median_s": float(df.t_tile.median())}
    L = ["# Faz 4G — uçtan uca hat testi (`guvenloc/pipeline.py`, `scripts/67_m3_pipeline.py`)", "",
         (f"{n_frames} kare (05 ve 11'den eşit aralıklı), {len(df)} çift (pozitif + kolay/zor negatif). Canlı: M1-C "
         "birleşimi (+TTA) → en olası dönme → RoMa v2 → RANSAC + hacim → S → conformal. Çevrimdışı: önbellek "
         "(cache/v1) + 64 satır 6. **Değerlendirme değil** (τ, q̂ aynı calib verisinden), tutarlılık testi."), "",
         "| ölçü | değer |", "|---|---|",
         f"| dönme uyuşması (canlı M1 vs 4a kararı) | {stats['rot_agree']:.3f} |",
         f"| kabul uyuşması (canlı dönme) | {stats['accept_agree']:.3f} |",
         f"| kabul uyuşması (aynı dönme) | {stats['accept_agree_same_rot']:.3f} |",
         f"| \\|ΔS\\| medyan / %95 (aynı dönme) | {stats['dS_median']:.4f} / {stats['dS_p95']:.4f} |",
         f"| n_inliers göreli fark medyan | {stats['n_inl_rel_diff_median']:.4f} |",
         f"| hacim tepesi kayması medyan / %95 | {stats['peak_shift_median_m']:.2f} / {stats['peak_shift_p95_m']:.2f} m |",
         f"| FAR-neg canlı / çevrimdışı | {stats['far_neg']:.3f} / {stats['far_neg_off']:.3f} |",
         f"| pozitif kabul | {stats['accept_pos']:.3f} |",
         f"| kapsama (kabul ∧ pozitif) | {stats['coverage']:.3f} |",
         f"| konum hatası medyan (kabul ∧ pozitif) | {stats['err_median_acc_pos_m']:.1f} m |",
         f"| enlem/boylam ↔ piksel hatası farkı medyan | {stats['latlon_vs_px_err_median_m']:.2f} m |",
         (f"| süre / çift medyan (karo + M1 + RoMa/hacim) | {stats['t_pair_median_s']:.2f} s "
         f"({stats['t_tile_median_s']:.2f} + {stats['t_m1_median_s']:.2f} + {stats['t_loc_median_s']:.2f}) |"), ""]
    (OUT / "e2e.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "e2e.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(L))
    log_run("67_m3_pipeline.py", Path("configs/h3.yaml"), 0, "e2e", "accept_agree", round(stats["accept_agree"], 5))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--e2e", type=int, default=0)
    a = ap.parse_args()
    h2 = yaml.safe_load(Path("configs/h2.yaml").read_text(encoding="utf-8"))
    h3 = yaml.safe_load(Path("configs/h3.yaml").read_text(encoding="utf-8"))
    pairs_all = pd.read_csv("data/pairs/pairs_v1.csv", dtype={"region": str, "tile_region": str})
    if a.export:
        export(h2, h3, pairs_all.set_index("pair_id"))
    if a.e2e:
        return e2e(h2, h3, pairs_all, a.e2e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
