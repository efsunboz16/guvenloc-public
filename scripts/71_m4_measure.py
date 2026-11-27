"""Faz 5B – filtre ölçümleri: her pozitif kare için tek kare kararı (satır 6) + θ üzerinden toplanmış hacim P(x, y).

  python scripts/71_m4_measure.py          # eğitim + calib (test 02, 03, 06: Faz 6)

Dönme: calib → M1-C birleşimi 4a kararı (configs/h3.yaml m1_model); eğitim → en çok inlier'lı dönme (satır 2; M1-C
eğitim bölgelerinde örneklem içi). S: calib → bütün eğitimde eğitilmiş h2_v3; eğitim → bölge-dışarıda (her bölge
kendisi olmadan). Kabul: S > τ (results/faz4/m3/m3_params_h3_v1.json, satır 6).
Hacim: önbellek cache/v1, configs/m2.yaml; P(x, y) = Σ_θ vol, 2 m hücreler 2×2 toplanıp 4 m'ye, toplam 1'e normalize.
→ results/faz5/m4/meas_<bölge>.npz (P: (kare, n, n) float16, en büyük karoya sağ/alt sıfır dolgulu) + meas_<bölge>.csv (kare tablosu)
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from guvenloc import m2_volume as V
from guvenloc.cache import pair_file
from guvenloc.matcher import MatchSet


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S61 = _load("m2s61", "scripts/61_m2_score.py")
S64 = _load("m3s64", "scripts/64_m3_conformal.py")
OUT = Path("results/faz5/m4")
TEST = {"02", "03", "06"}
TRAIN = ["01", "04", "08", "09", "10"]
POOL = 2                                       # 2 m → 4 m


def frame_table(h2, h3, pairs) -> pd.DataFrame:
    tr, ca = S61.load_split("train", pairs), S61.load_split("calib", pairs)
    feats = h2["score_model"]["features"]
    ca["S"] = S61.predict(S64.fit_S(tr, h2), S61.feat_matrix(ca, feats))
    tr["S"] = np.nan
    for r in TRAIN:
        m = tr.region == r
        tr.loc[m, "S"] = S61.predict(S64.fit_S(tr[~m], h2), S61.feat_matrix(tr[m], feats))
    dec = pd.read_csv(S61.M1_DECISIONS.format(model=h3["system"]["m1_model"]))
    ca_f = S64.frame_rows(ca, dec[dec.row == "4a"].set_index("pair_id").rot)
    tr_f = tr.loc[tr.groupby("pair_id").n_inliers.idxmax()]
    d = pd.concat([tr_f, ca_f], ignore_index=True)
    return d[d.is_pos].reset_index(drop=True)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    h2 = yaml.safe_load(Path("configs/h2.yaml").read_text(encoding="utf-8"))
    h3 = yaml.safe_load(Path("configs/h3.yaml").read_text(encoding="utf-8"))
    params = json.loads(Path("results/faz4/m3/m3_params_h3_v1.json").read_text(encoding="utf-8"))
    pairs = pd.read_csv("data/pairs/pairs_v1.csv", dtype={"region": str, "tile_region": str}).set_index("pair_id")
    d = frame_table(h2, h3, pairs)
    assert not TEST & set(d.region)
    measure(d, pairs, params, verbose=True)
    return 0


def measure(d: pd.DataFrame, pairs: pd.DataFrame, params: dict, verbose: bool) -> None:
    """Kare tablosu (pozitif çiftler, seçilen dönme, S, ölçümler) → meas_<bölge>.npz/.csv.
    verbose=False (Faz 6 testi): hata özeti yazdırılmaz."""
    d = d.drop(columns=[c for c in ("tile_size_m", "tile_px", "gt_x", "gt_y", "label") if c in d]).join(
        pairs[["drone_path", "tile_lat", "tile_lon", "tile_size_m", "tile_px", "scale_s", "gt_x", "gt_y",
               "drone_lat", "drone_lon", "drone_m_per_px", "gt_heading", "terrain"]], on="pair_id")
    d["accept"] = d.S > params["tau"]
    d["sigma_hat"] = np.maximum(d.spread_m.fillna(50.0), params["sigma_floor_m"])
    d["num"] = d.pair_id.str.split("_").str[1].astype(int)
    vcfg = V.VolumeCfg(**params["volume_cfg"])
    OUT.mkdir(parents=True, exist_ok=True)
    for rid, g in d.sort_values(["region", "num"]).groupby("region"):
        Ps, t0 = [], time.time()
        for r in g.itertuples():
            ms = MatchSet.load_npz(pair_file("cache/v1", r.pair_id, int(r.rot)))
            mpp = r.tile_size_m / r.tile_px
            vol = V.build_volume(ms.u, ms.v, ms.c, ms.Sigma_B, ms.Sigma_A, float(r.scale_s), ms.meta["drone_wh"],
                                 int(r.tile_px), mpp, vcfg)["vol"]
            P = vol.sum(0)
            n = P.shape[0] - P.shape[0] % POOL
            P = P[:n, :n].reshape(n // POOL, POOL, n // POOL, POOL).sum((1, 3))
            P = P / P.sum().clamp_min(1e-12)
            Ps.append(P.to(torch.float16).cpu().numpy())
        m = max(x.shape[0] for x in Ps)                    # karo boyu kareye göre değişir: sağ/alt sıfır dolgu (= dışarı)
        np.savez_compressed(OUT / f"meas_{rid}.npz",
                            P=np.stack([np.pad(x, ((0, m - x.shape[0]), (0, m - x.shape[1]))) for x in Ps]))
        g = g.assign(cell_m=vcfg.cell_m * POOL)
        g.drop(columns=[c for c in g.columns if c.startswith("block")]).to_csv(OUT / f"meas_{rid}.csv", index=False)
        print(f"{rid}: {len(g)} kare, P {Ps[0].shape}, {time.time() - t0:.0f} s"
              + (f"; kabul {g.accept.mean():.3f}, tek kare medyan hata {g.err_m.median():.0f} m" if verbose else ""),
              flush=True)


if __name__ == "__main__":
    sys.exit(main())
