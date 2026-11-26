"""Faz 2A – Modül 1 yön etiketleri ve etiket gürültüsü kontrolü (GPU gerekmez; cache/v1'den).

  python scripts/50_m1_labels.py --cache cache/v1

Etiket (MIMARI 2A): φ = north_up_rotation_cw_deg(Phi1) (= Phi1 mod 360; görüntü üst kenarının pusula
yönü, Faz 1B'de doğrulandı), bölme b(φ) = floor(φ / 10°) mod 36.

Gürültü kontrolü: RoMa v2 homografisinin verdiği görüntü-üstü pusula yönü (satır 2 — 4 yön + en çok
inlier — KABUL ve konum hatası < 25 m olan kareler, yani eşleşmesi güvenilir kareler) ile φ arasındaki
fark d = wrap(θ_RoMa − φ) ∈ (−180°, 180°]. Bölge ve uçuş bacağı (Phi1'in ardışık karelerde > 30°
değiştiği yerlerden bölünen dilimler) başına raporlanır.

Doğrulama uçuşu (karar 26.11, kural önceden konmuştur): eğitim bölgelerinden, coğrafi grubunu başka bir
eğitim bölgesiyle paylaşmayan ve ≥ 500 karesi olan adaylar içinden medyan |d|'si en küçük olan.
M1'in yön hatası calib bölgelerinde (05, 11) ölçülür (karar 26.11); calib etiketleri eğitimde kullanılmaz.
Test bölgelerine dokunulmaz.

Düzeltme (karar F2-4, 26.11): gürültü uçuş yönüne bağlı ve sistematik olduğundan (Phi1 rota yönü + burun
açısı), train/val bölgelerinde bölge × uçuş yönü başına sabit c = medyan d eklenir → φ_corr (labels_v2).

Çıktılar: data/m1/labels_v1.csv, data/m1/labels_v2.csv (+ docs/checksums/m1_labels_v*.sha256),
results/faz2/label_noise.{md,json}, results/faz2/label_correction.{md,json}.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc.geo import north_up_rotation_cw_deg
from guvenloc.metrics import log_run
from guvenloc.pairs import load_split

OUT = Path("results/faz2")
LABELS = Path("data/m1/labels_v1.csv")
LABELS_V2 = Path("data/m1/labels_v2.csv")
DIR_TOL_DEG = 45.0
MIN_CORR_N = 20
N_BINS = 36
LEG_JUMP_DEG = 30.0
MAX_ERR_M = 25.0
VAL_MIN_FRAMES = 500


def wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def _b40():
    spec = importlib.util.spec_from_file_location("b40", Path(__file__).with_name("40_baselines.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def legs(phi1: np.ndarray) -> np.ndarray:
    """Phi1 ardışık karelerde > 30° değiştiğinde yeni bacak."""
    jump = np.abs(wrap180(np.diff(phi1))) > LEG_JUMP_DEG
    return np.concatenate([[0], np.cumsum(jump)])


def noise_stats(d: pd.Series) -> dict:
    a = d.abs().dropna()
    if not len(a):
        return {"n_reliable": 0}
    return {"n_reliable": len(a), "median_d": float(np.median(d.dropna())),
            "median_abs_d": float(np.median(a)), "p90_abs_d": float(np.quantile(a, 0.9)),
            "frac_gt10": float((a > 10).mean()), "frac_gt20": float((a > 20).mean())}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("cache/v1"))
    ap.add_argument("--pairs", type=Path, default=Path("data/pairs/pairs_v1.csv"))
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    sel = yaml.safe_load(Path("configs/baselines.yaml").read_text(encoding="utf-8"))
    px, taus = sel["ransac_px"], {int(k): v for k, v in sel["tau_inliers"].items()}
    split = load_split()
    pairs = pd.read_csv(a.pairs, dtype={"region": str, "tile_region": str})
    b40 = _b40()

    rows = []
    for sp in ("train", "calib"):
        pos = pairs[(pairs.split == sp) & (pairs.label == "pos")].copy()
        feats = pd.read_csv(Path("results/faz1/baselines") / f"feats_{a.cache.name}_{sp}_px{px:g}.csv")
        r = b40.evaluate_rows(feats, pairs[pairs.split == sp], taus)
        r2 = r[(r.row == 2) & r.is_pos].set_index("pair_id")
        best = feats.loc[feats.groupby("pair_id").n_inliers.idxmax()].set_index("pair_id")
        pos = pos.set_index("pair_id")
        pos["num"] = pos.index.str.split("_").str[1].astype(int)
        pos = pos.sort_values(["region", "num"])
        pos["phi"] = [north_up_rotation_cw_deg(p) for p in pos.gt_heading]
        pos["bin"] = (np.floor(pos.phi / (360.0 / N_BINS)).astype(int)) % N_BINS
        pos["leg"] = pos.groupby("region").gt_heading.transform(lambda s: legs(s.to_numpy()))
        ok = r2.accept & (r2.err_m < MAX_ERR_M)
        pos["reliable"] = ok.reindex(pos.index).fillna(False).astype(bool)
        pos["roma_bearing"] = best.bearing.reindex(pos.index).where(pos.reliable)
        pos["d"] = wrap180(pos.roma_bearing - pos.phi)
        rows.append(pos)
    lab = pd.concat(rows)

    # bölge ve bacak özetleri
    reg = {rid: {"split": g.split.iloc[0], "n_frames": len(g), "group": split.group_of[rid],
                 "n_legs": int(g.leg.nunique()), **noise_stats(g.d)} for rid, g in lab.groupby("region")}
    leg_rows = []
    for (rid, leg), g in lab.groupby(["region", "leg"]):
        s = noise_stats(g.d)
        leg_rows.append({"region": rid, "leg": int(leg), "n": len(g), "phi_median": float(np.median(g.phi)),
                         **s})
    legs_df = pd.DataFrame(leg_rows)

    # doğrulama uçuşu (önceden konmuş kural)
    train = [r for r in split.regions("train")]
    shared = {r for r in train if sum(split.group_of[x] == split.group_of[r] for x in train) > 1}
    cands = [r for r in train if r not in shared and reg[r]["n_frames"] >= VAL_MIN_FRAMES]
    val = min(cands, key=lambda r: reg[r]["median_abs_d"])
    lab["m1_role"] = np.where(lab.split == "calib", "eval",
                              np.where(lab.region == val, "val", "train"))

    LABELS.parent.mkdir(parents=True, exist_ok=True)
    cols = ["region", "split", "m1_role", "num", "drone_path", "gt_heading", "phi", "bin", "leg",
            "reliable", "roma_bearing", "d"]
    lab.reset_index()[["pair_id", *cols]].to_csv(LABELS, index=False, float_format="%.4f")
    digest = hashlib.sha256(LABELS.read_bytes()).hexdigest()
    ck = Path("docs/checksums/m1_labels_v1.sha256")
    ck.write_text(f"{digest}  {LABELS.as_posix()}\n", encoding="utf-8")

    tr = lab[lab.split == "train"]
    res = {"labels": str(LABELS), "sha256": digest, "n_bins": N_BINS,
           "train_all": noise_stats(tr.d), "calib_all": noise_stats(lab[lab.split == "calib"].d),
           "by_region": reg, "val_rule": f"grup paylaşmayan, ≥ {VAL_MIN_FRAMES} kare, en küçük medyan |d|",
           "val_candidates": {r: reg[r]["median_abs_d"] for r in cands}, "val_region": val,
           "roles": lab.m1_role.value_counts().to_dict(),
           "bin_hist_train": np.bincount(lab[lab.m1_role == "train"].bin, minlength=N_BINS).tolist(),
           "noisy_legs": legs_df[(legs_df.n_reliable >= 10) & (legs_df.median_abs_d > 10)].to_dict("records")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "label_noise.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    legs_df.to_csv(OUT / "label_noise_legs.csv", index=False, float_format="%.3f")

    L = ["# Faz 2A — M1 yön etiketleri ve etiket gürültüsü", "",
         (f"Etiket: φ = Phi1 mod 360, {N_BINS} bölme (10°). Gürültü: d = θ_RoMa − φ, yalnızca satır 2 kabul + "
         f"konum hatası < {MAX_ERR_M:g} m olan (güvenilir) karelerde. `{LABELS}` sha256 `{digest[:16]}…`."), "",
         "| bölge | küme | kare | bacak | güvenilir | medyan d | medyan \\|d\\| | p90 \\|d\\| | \\|d\\| > 10° | \\|d\\| > 20° |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for rid, s in reg.items():
        if not s.get("n_reliable"):
            L.append(f"| {rid} | {s['split']} | {s['n_frames']} | {s['n_legs']} | 0 | – | – | – | – | – |")
            continue
        L.append(f"| {rid} | {s['split']} | {s['n_frames']} | {s['n_legs']} | {s['n_reliable']} | "
                 f"{s['median_d']:+.1f}° | {s['median_abs_d']:.1f}° | {s['p90_abs_d']:.1f}° | "
                 f"%{s['frac_gt10'] * 100:.0f} | %{s['frac_gt20'] * 100:.0f} |")
    t = res["train_all"]
    L += ["", (f"Eğitim toplam: {t['n_reliable']} güvenilir kare, medyan d {t['median_d']:+.1f}°, "
          f"medyan |d| {t['median_abs_d']:.1f}°, |d| > 10° %{t['frac_gt10'] * 100:.0f}."), "",
          f"**Doğrulama uçuşu (kural: {res['val_rule']}):** adaylar "
          + ", ".join(f"{r} ({v:.1f}°)" for r, v in res["val_candidates"].items())
          + f" → **{val}**. Roller: " + ", ".join(f"{k} {v}" for k, v in res["roles"].items()) + ".", "",
          f"Gürültülü bacaklar (≥ 10 güvenilir kare, medyan |d| > 10°): {len(res['noisy_legs'])}"]
    for lg in res["noisy_legs"]:
        L.append(f"- {lg['region']} bacak {lg['leg']}: {lg['n']} kare, φ ≈ {lg['phi_median']:.0f}°, "
                 f"medyan d {lg['median_d']:+.1f}°, medyan |d| {lg['median_abs_d']:.1f}°")
    (OUT / "label_noise.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    seed = int(cfg["seed"])
    log_run("50_m1_labels.py", a.config, seed, "train", "label_noise_median_abs_d", round(t["median_abs_d"], 3),
            notes=f"n={t['n_reliable']} val={val}")
    log_run("50_m1_labels.py", a.config, seed, "train", "label_noise_frac_gt10", round(t["frac_gt10"], 4))
    correct_labels(lab, a.config, seed)
    return 0


def directions(leg_phi: pd.Series) -> dict[int, int]:
    """Bacakları uçuş yönüne göre kümeler: medyan φ'leri ≤ 45° (dairesel) olan bacaklar aynı yön."""
    centers: list[float] = []
    out = {}
    for leg, p in leg_phi.items():
        for k, c in enumerate(centers):
            if abs(wrap180(p - c)) <= DIR_TOL_DEG:
                out[leg] = k
                break
        else:
            centers.append(p)
            out[leg] = len(centers) - 1
    return out


def correct_labels(lab: pd.DataFrame, config: Path, seed: int) -> None:
    """Karar F2-4 (26.11): bölge × uçuş yönü başına sabit düzeltme (yalnızca train/val rolleri).

    düzeltme c(bölge, yön) = o yöndeki güvenilir karelerin medyan d'si (n ≥ MIN_CORR_N; değilse 0, etiket
    ham kalır). φ_corr = φ + c. Calib (eval) etiketleri değişmez. Düzeltmenin dürüst etkisi birini-dışarıda-
    bırak ile ölçülür: her bacağın düzeltmesi yalnızca aynı yöndeki DİĞER bacaklardan hesaplanıp o bacakta sınanır.
    """
    lab = lab.copy()
    lab["dir"] = -1
    lab["corr_deg"] = 0.0
    lab["corr_n"] = 0
    lolo = []
    for rid, g in lab[lab.m1_role != "eval"].groupby("region"):
        leg_phi = g.groupby("leg").phi.median()
        dmap = directions(leg_phi)
        lab.loc[g.index, "dir"] = g.leg.map(dmap).to_numpy()
        gg = lab.loc[g.index]
        for k, h in gg.groupby("dir"):
            dd = h.d.dropna()
            if len(dd) >= MIN_CORR_N:
                lab.loc[h.index, "corr_deg"] = float(np.median(dd))
                lab.loc[h.index, "corr_n"] = len(dd)
            for leg, hl in h.groupby("leg"):
                own = hl.d.dropna()
                other = h[h.leg != leg].d.dropna()
                if len(own) < 5 or len(other) < MIN_CORR_N:
                    continue
                c = float(np.median(other))
                lolo.append({"region": rid, "dir": int(k), "leg": int(leg), "n": len(own),
                             "abs_before": np.abs(own.to_numpy()), "abs_after": np.abs(wrap180(own - c))})
    lab["phi_corr"] = (lab.phi + lab.corr_deg) % 360.0
    lab["bin_corr"] = (np.floor(lab.phi_corr / (360.0 / N_BINS)).astype(int)) % N_BINS

    LABELS_V2.parent.mkdir(parents=True, exist_ok=True)
    cols = ["pair_id", "region", "split", "m1_role", "num", "drone_path", "gt_heading", "phi", "bin", "leg",
            "dir", "corr_deg", "corr_n", "phi_corr", "bin_corr", "reliable", "roma_bearing", "d"]
    lab.reset_index()[cols].to_csv(LABELS_V2, index=False, float_format="%.4f")
    digest = hashlib.sha256(LABELS_V2.read_bytes()).hexdigest()
    Path("docs/checksums/m1_labels_v2.sha256").write_text(f"{digest}  {LABELS_V2.as_posix()}\n", encoding="utf-8")

    corr_tab = (lab[lab.m1_role != "eval"].groupby(["region", "dir"])
                .agg(n=("phi", "size"), phi_med=("phi", "median"), corr=("corr_deg", "first"), corr_n=("corr_n", "first"))
                .reset_index())
    L = ["# Faz 2A — etiket düzeltmesi (karar F2-4)", "",
         (f"Bölge × uçuş yönü başına c = güvenilir karelerin medyan d'si (n ≥ {MIN_CORR_N}; değilse c = 0). "
         f"φ_corr = φ + c. Yalnızca train/val; calib değişmez. `{LABELS_V2}` sha256 `{digest[:16]}…`."), "",
         "| bölge | yön | kare | medyan φ | c | c'nin dayandığı kare |", "|---|---|---|---|---|---|"]
    for r in corr_tab.itertuples():
        L.append(f"| {r.region} | {r.dir} | {r.n} | {r.phi_med:.0f}° | "
                 + (f"{r.corr:+.1f}°" if r.corr_n else "0 (ham; n az)") + f" | {r.corr_n} |")
    res = {"labels_v2": str(LABELS_V2), "sha256": digest, "min_corr_n": MIN_CORR_N,
           "corrections": corr_tab.to_dict("records"), "lolo": {}}
    L += ["", "**Birini-dışarıda-bırak (bacağın düzeltmesi yalnızca aynı yöndeki diğer bacaklardan):**", "",
          "| bölge | sınanan bacak | güvenilir kare | medyan \\|d\\| önce | sonra | \\|d\\| > 10° önce | sonra |",
          "|---|---|---|---|---|---|---|"]
    groups = {}
    for x in lolo:
        groups.setdefault(x["region"], []).append(x)
    groups["hepsi"] = lolo
    for rid, xs in groups.items():
        if not xs:
            continue
        b = np.concatenate([x["abs_before"] for x in xs])
        af = np.concatenate([x["abs_after"] for x in xs])
        s = {"legs": len(xs), "n": len(b), "median_abs_before": float(np.median(b)),
             "median_abs_after": float(np.median(af)), "frac_gt10_before": float((b > 10).mean()),
             "frac_gt10_after": float((af > 10).mean())}
        res["lolo"][rid] = s
        L.append(f"| {rid} | {s['legs']} | {s['n']} | {s['median_abs_before']:.1f}° | {s['median_abs_after']:.1f}° | "
                 f"%{s['frac_gt10_before'] * 100:.0f} | %{s['frac_gt10_after'] * 100:.0f} |")
    (OUT / "label_correction.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float),
                                               encoding="utf-8")
    (OUT / "label_correction.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L))
    h = res["lolo"].get("hepsi")
    if h:
        log_run("50_m1_labels.py", config, seed, "train", "label_lolo_median_abs_after", round(h["median_abs_after"], 3),
                notes=f"önce {h['median_abs_before']:.3f}, n={h['n']}")


if __name__ == "__main__":
    sys.exit(main())
