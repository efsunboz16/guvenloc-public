"""Faz 6 – test bölgeleri (02, 03, 06) için girdi hazırlığı. DEĞERLENDİRME YOK: hiçbir hata/başarı yazdırılmaz.

  python scripts/80_final_prep.py            # eksik adımları sırayla üretir (var olan çıktı atlanır)

Adımlar (hepsi calib'deki karşılıklarıyla aynı kod):
  feats  40_baselines feats (RANSAC, 5 px)              → results/faz1/baselines/feats_v1_test_px5.csv
  vol    60_m2_features ile aynı hacim metrikleri        → results/faz3/m2/vol_feats_v1_test.csv
  m1     M1-C birleşimi (+TTA) dönme olasılıkları       → results/faz6/prep/m1_probs_test.npz
  seq    drone↔drone hareket (70_m4_sequences.seq_motion) → results/faz6/prep/seq_motion_test.csv
  meas   filtre ölçümleri (71_m4_measure.measure; dönme = M1-C en olası, S = h2_v3, kabul = τ)
                                                          → results/faz5/m4/meas_{02,03,06}.npz/.csv
Etiketler (gt) yalnız calib'deki dosya şemasını korumak için sütun olarak hesaplanır, özetlenmez.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import itertools

from guvenloc import m2_volume as V
from guvenloc.cache import pair_file, seq_file

PREP = Path("results/faz6/prep")
ROTS = (0, 90, 180, 270)
TEST = ["02", "03", "06"]


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def step_feats() -> None:
    out = Path("results/faz1/baselines/feats_v1_test_px5.csv")
    if out.exists():
        return
    subprocess.run([sys.executable, "scripts/40_baselines.py", "feats", "--cache", "cache/v1", "--split", "test",
                    "--ransac-px", "5", "--workers", "12"], check=True)


def step_vol(pairs: pd.DataFrame) -> None:
    out = Path("results/faz3/m2/vol_feats_v1_test.csv")
    if out.exists():
        return
    base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    mcfg = yaml.safe_load(Path("configs/m2.yaml").read_text(encoding="utf-8"))
    pp = pairs[pairs.split == "test"].set_index("pair_id")
    jobs = [(pid, r) for pid in pp.index for r in ROTS]

    def load(job):
        pid, rot = job
        z = np.load(pair_file("cache/v1", pid, rot))
        return pid, rot, {k: z[k] for k in ("u", "v", "c", "Sigma_A", "Sigma_B")}, json.loads(str(z["meta"]))

    rows, t0 = [], time.time()
    with ThreadPoolExecutor(6) as ex:
        for k, (pid, rot, z, meta) in enumerate(ex.map(load, jobs)):
            m = pp.loc[pid]
            cfg = V.VolumeCfg(**mcfg, sigma_scale=float(base["romav2"]["sigma_scale"]) / float(meta["sigma_scale"]))
            mpp = float(m.tile_size_m) / float(m.tile_px)
            r = V.metrics(V.build_volume(z["u"], z["v"], z["c"], z["Sigma_B"], z["Sigma_A"], float(m.scale_s),
                                         meta["drone_wh"], int(m.tile_px), mpp, cfg))
            err = float(np.hypot(r["x"] - m.gt_x, r["y"] - m.gt_y) * mpp) if m.label == "pos" else np.nan
            herr = float(abs((r["theta"] - m.gt_heading + 180) % 360 - 180)) if m.label == "pos" else np.nan
            rows.append({"pair_id": pid, "rot": rot, **r, "err_m": err, "heading_err": herr})
            if (k + 1) % 2000 == 0:
                el = time.time() - t0
                print(f"vol: {k + 1}/{len(jobs)} ({el / (k + 1) * 1000:.0f} ms/dosya, kalan "
                      f"{el / (k + 1) * (len(jobs) - k - 1) / 60:.0f} dk)", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"vol: {len(rows)} satır yazıldı", flush=True)


def step_m1(pairs: pd.DataFrame, model: str) -> None:
    out = PREP / "m1_probs_test.npz"
    if out.exists():
        return
    R56 = _load("m1c56", "scripts/56_m1c_roma.py")
    pt = pairs[pairs.split == "test"].sort_values("pair_id").reset_index(drop=True)
    tiles = R56.tiles_for(pt, Path("data/m1/test_pairs_513.npy"))
    probs, phi = R56.m1_rot_probs(model, pt, tiles)
    np.savez(out, pair_id=pt.pair_id.to_numpy(), probs=probs, phi=phi)
    print(f"m1: {len(pt)} çift", flush=True)


def step_seq(pairs: pd.DataFrame) -> None:
    out = PREP / "seq_motion_test.csv"
    if out.exists():
        return
    S70 = _load("m4s70", "scripts/70_m4_sequences.py")
    pos = pairs[(pairs.label == "pos") & pairs.region.isin(TEST)].sort_values(["region", "drone_path"])
    rows = []
    for rid, g in pos.groupby("region"):
        nums = g.pair_id.str.split("_").str[1].astype(int).tolist()
        jobs = list(itertools.pairwise(nums))
        with ThreadPoolExecutor(8) as ex:
            mots = list(ex.map(S70.seq_motion, [seq_file("cache/v1", rid, a, b) for a, b in jobs]))
        rows += [{"region": rid, "t0": a, "t1": b, **m} for (a, b), m in zip(jobs, mots, strict=True)]
        print(f"seq: {rid} {len(jobs)} ardışık çift", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def step_meas(pairs: pd.DataFrame, h2: dict) -> None:
    if all(Path(f"results/faz5/m4/meas_{r}.npz").exists() for r in TEST):
        return
    S61 = _load("m2s61", "scripts/61_m2_score.py")
    S64 = _load("m3s64", "scripts/64_m3_conformal.py")
    S71 = _load("m4s71", "scripts/71_m4_measure.py")
    params = json.loads(Path("results/faz4/m3/m3_params_h3_v1.json").read_text(encoding="utf-8"))
    pi = pairs.set_index("pair_id")
    tr, ts = S61.load_split("train", pi), S61.load_split("test", pi)
    ts["S"] = S61.predict(S64.fit_S(tr, h2), S61.feat_matrix(ts, h2["score_model"]["features"]))
    z = np.load(PREP / "m1_probs_test.npz", allow_pickle=True)
    rot = pd.Series([ROTS[i] for i in z["probs"].argmax(1)], index=z["pair_id"])
    d = S64.frame_rows(ts, rot)
    d = d[d.is_pos].reset_index(drop=True)
    S71.measure(d, pi, params, verbose=False)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    PREP.mkdir(parents=True, exist_ok=True)
    h2 = yaml.safe_load(Path("configs/h2.yaml").read_text(encoding="utf-8"))
    h3 = yaml.safe_load(Path("configs/h3.yaml").read_text(encoding="utf-8"))
    pairs = pd.read_csv("data/pairs/pairs_v1.csv", dtype={"region": str, "tile_region": str})
    t0 = time.time()
    for name, fn in (("feats", lambda: step_feats()), ("vol", lambda: step_vol(pairs)),
                     ("m1", lambda: step_m1(pairs, h3["system"]["m1_model"])), ("seq", lambda: step_seq(pairs)),
                     ("meas", lambda: step_meas(pairs, h2))):
        print(f"[{time.time() - t0:.0f} s] adım {name}", flush=True)
        fn()
    print(f"HAZIRLIK_BITTI ({time.time() - t0:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
