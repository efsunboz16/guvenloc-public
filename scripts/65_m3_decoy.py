"""Faz 4, karar F4-4 – tuzak karo (decoy) normalizasyonu: kare başına sahte eşleşme düzeyi.

  python scripts/65_m3_decoy.py            # benzetim (F4-4): YALNIZ eğitim bölgeleri, bölge-dışarıda
  python scripts/65_m3_decoy.py --real     # gerçek tuzak çiftleri (F4-5; 66_m3_decoy_pairs.py + önbellek)

Kuru denemede (F4-3) FAR-neg bölgeler arası taşınmadı: negatiflerin sahte inlier sayısı bölgeye bağlı (05 ≫ 11).
Öneri: her kareyi bilinen yanlış bir karoyla da eşleştir, skoru farkla normalize et:
    n_norm = log1p(n) − log1p(n_tuzak),   S_norm = S − S(tuzak)
Benzetim (önbellekten, yeni RoMa yok): karenin ÜÇ çiftine de (pos, easy0, hard0) AYNI tuzak = komşu karenin
(aynı bölge, indeks +1, yoksa −1) easy0 çifti, değerlendirilen çiftin dönmesinde. Böylece tuzak hiçbir çiftte
kendisi değil ve çift türüne göre değişmiyor (yansızlık). Dağıtımda tuzak = rastgele uzak bir karo, aynı kare.

Dönme: her çiftte 4 dönmenin en çok inlier'lısı (satır 2; M1-C eğitim bölgelerinde örneklem içi olduğu için).
Her dışarıda bırakılan bölge r için: S diğer 4 bölgede eğitilir, τ diğer 4 bölgenin negatiflerinden conformal
(α = 0,05) → FAR_r ve doğru pozitif kabulü.

Seçim kuralı (sonuçtan önce): bir aile (inlier, S) için normalize sürüm, ham sürüme göre ortalama |FAR_r − α|
ve en büyük FAR_r'yi birlikte düşürür ve havuzlanmış bölge-dışarıda AUROC'u 0,005'ten fazla düşürmezse
"geçer". Geçenler arasında ortalama |FAR_r − α|'sı en küçük olan seçilir; hiçbiri geçmezse ham S kalır.
→ results/faz4/m3/decoy_sel.json / .md
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
from guvenloc import metrics as M


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S61 = _load("m2s61", "scripts/61_m2_score.py")
OUT = Path("results/faz4/m3")
ALPHA = 0.05
TRAIN = ["01", "04", "08", "09", "10"]


def with_real_decoy(d: pd.DataFrame, dec: pd.DataFrame) -> pd.DataFrame:
    """F4-5: tuzak = aynı karenin kendi tuzak çifti (<kare>_decoy0, 66_m3_decoy_pairs.py), aynı dönme."""
    d = d.copy()
    d["decoy_pid"] = d.pair_id.str.rsplit("_", n=1).str[0] + "_decoy0"
    dz = dec[["pair_id", "rot", "n_inliers", "S"]].rename(columns=lambda c: c if c == "rot" else c + "_d")
    d = d.merge(dz, left_on=["decoy_pid", "rot"], right_on=["pair_id_d", "rot"], how="left")
    return d[d.pair_id_d.notna()]


def with_decoy(d: pd.DataFrame) -> pd.DataFrame:
    """(çift, dönme) satırlarına komşu karenin easy0 çiftinin aynı dönmedeki özniteliklerini ekler (_d eki)."""
    d = d.copy()
    parts = d.pair_id.str.split("_")
    d["frame"] = parts.str[0] + "_" + parts.str[1]
    d["fidx"] = parts.str[1].astype(int)
    easy = d[parts.str[2] == "easy0"]
    have = set(zip(easy.region, easy.fidx, strict=True))
    nb = []
    for reg, i in zip(d.region, d.fidx, strict=True):
        nb.append(f"{reg}_{i + 1:04d}" if (reg, i + 1) in have else f"{reg}_{i - 1:04d}" if (reg, i - 1) in have
                  else None)
    d["decoy_pid"] = [f"{f}_easy0" if f else None for f in nb]
    cols = ["pair_id", "rot", "n_inliers", "S"]
    dz = easy[cols].rename(columns=lambda c: c if c == "rot" else c + "_d")
    d = d.merge(dz, left_on=["decoy_pid", "rot"], right_on=["pair_id_d", "rot"], how="left")
    return d[d.pair_id_d.notna()]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    h2 = yaml.safe_load(Path("configs/h2.yaml").read_text(encoding="utf-8"))
    sm = h2["score_model"]
    real = "--real" in sys.argv
    pairs = pd.read_csv("data/pairs/pairs_v1.csv", dtype={"region": str}).set_index("pair_id")
    tr = S61.load_split("train", pairs)
    dec = None
    if real:
        dp = pd.read_csv("data/pairs/decoy_v1.csv", dtype={"region": str}).set_index("pair_id")
        dec = S61.load_split("train", dp, tag="decoy_v1")
        dec["S"] = np.nan
    tr["S"] = np.nan
    for r in TRAIN:                                         # S: bölge-dışarıda (her bölge kendisi olmadan)
        m = tr.region == r
        model = S61.fit_logistic(S61.feat_matrix(tr[~m], sm["features"]), tr[~m].ok_ransac.to_numpy(float),
                                 float(sm["C"]))
        tr.loc[m, "S"] = S61.predict(model, S61.feat_matrix(tr[m], sm["features"]))
        if real:
            md = dec.region == r
            dec.loc[md, "S"] = S61.predict(model, S61.feat_matrix(dec[md], sm["features"]))
    d = with_real_decoy(tr, dec) if real else with_decoy(tr)
    best = d.loc[d.groupby("pair_id").n_inliers.idxmax()].reset_index(drop=True)   # satır 2 dönmesi
    li, lid = np.log1p(best.n_inliers.clip(lower=0)), np.log1p(best.n_inliers_d.clip(lower=0))
    scores = {"inlier": li.to_numpy(), "inlier_norm": (li - lid).to_numpy(),
              "S": best.S.to_numpy(), "S_norm": (best.S - best.S_d).to_numpy()}
    ok, pos = best.ok_ransac.to_numpy(), best.is_pos.to_numpy()
    res = {}
    for name, sc in scores.items():
        far, pw = {}, {}
        for r in TRAIN:
            m = (best.region == r).to_numpy()
            tau = C.reject_threshold(sc[~m & ~pos], ALPHA)
            far[r] = float((sc[m & ~pos] > tau).mean())
            pw[r] = float((sc[m & pos & ok] > tau).mean())
        dev = [abs(v - ALPHA) for v in far.values()]
        res[name] = {"far": far, "accept_correct_pos": pw, "mean_abs_dev": float(np.mean(dev)),
                     "max_far": float(max(far.values())), "auroc": M.auroc(sc, ok)}
    sel, passed = "S", []
    for fam in ("inlier", "S"):
        raw, nor = res[fam], res[fam + "_norm"]
        if (nor["mean_abs_dev"] < raw["mean_abs_dev"] and nor["max_far"] < raw["max_far"]
                and nor["auroc"] >= raw["auroc"] - 0.005):
            passed.append(fam + "_norm")
    if passed:
        sel = min(passed, key=lambda k: res[k]["mean_abs_dev"])
    kind = ("gerçek tuzak: aynı kare × başka bölgeden karo (F4-5)" if real
            else "benzetim: komşu karenin easy0 çifti (F4-4)")
    lines = [f"# Faz 4 — tuzak karo normalizasyonu, seçim (eğitim bölgeleri, bölge-dışarıda) — {kind}", "",
             f"Çift sayısı {len(best)} (tuzağı olan kareler), α = {ALPHA}, aynı dönme.", "",
             "| skor | " + " | ".join(f"FAR {r}" for r in TRAIN) + " | ort. \\|FAR − α\\| | en büyük FAR | AUROC | "
             "doğru poz. kabul (ort.) |", "|---|" + "---|" * (len(TRAIN) + 4)]
    for name, v in res.items():
        lines.append(f"| {name} | " + " | ".join(f"{v['far'][r]:.3f}" for r in TRAIN) +
                     f" | {v['mean_abs_dev']:.3f} | {v['max_far']:.3f} | {v['auroc']:.4f} | "
                     f"{np.mean(list(v['accept_correct_pos'].values())):.3f} |")
    lines += ["", f"Kuralı geçen: {passed or 'yok'} → **seçilen: {sel}**"]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_real" if real else ""
    (OUT / f"decoy_sel{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / f"decoy_sel{tag}.json").write_text(json.dumps({"results": res, "passed": passed, "selected": sel},
                                                   indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
