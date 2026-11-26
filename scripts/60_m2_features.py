"""Faz 3A–3B – her (çift, dönme) için oylama hacmi metrikleri → results/faz3/m2/vol_feats_v1_<split>.csv.

  python scripts/60_m2_features.py --splits train calib        # test (02, 03, 06) Faz 6'ya kadar kapalı
  python scripts/60_m2_features.py --splits calib --limit 200  # hızlı deneme

Girdi: cache/v1/<pair_id>_r<rot>.npz (INTERFACES §1), data/pairs/pairs_v1.csv (scale_s, tile_px, tile_size_m).
Σ ölçeği: configs/base.yaml romav2.sigma_scale / meta.sigma_scale. Hacim: configs/m2.yaml (θ M1'den bağımsız).
Konum hatası (pozitif): |tepe − GT| m. Satırlar faz1 feats_v1 ile aynı anahtarda (pair_id, rot).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc import m2_volume as V
from guvenloc.cache import pair_file
from guvenloc.metrics import log_run

ROTS = (0, 90, 180, 270)
OUT = Path("results/faz3/m2")


CACHE = {"dir": "cache/v1"}


def load(job):
    pid, rot = job
    z = np.load(pair_file(CACHE["dir"], pid, rot))
    meta = json.loads(str(z["meta"]))
    return pid, rot, {k: z[k] for k in ("u", "v", "c", "Sigma_A", "Sigma_B")}, meta


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "calib"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pairs", default="data/pairs/pairs_v1.csv")
    ap.add_argument("--cache", default="cache/v1")
    ap.add_argument("--tag", default="v1", help="çıktı: vol_feats_<tag>_<split>.csv (tuzak: decoy_v1)")
    a = ap.parse_args()
    CACHE["dir"] = a.cache
    if "test" in a.splits:
        sys.exit("test bölmesi Faz 6'ya kadar kapalı")
    base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    mcfg = yaml.safe_load(Path("configs/m2.yaml").read_text(encoding="utf-8"))
    pairs = pd.read_csv(a.pairs, dtype={"region": str, "tile_region": str}).set_index("pair_id")
    for split in a.splits:
        pp = pairs[pairs.split == split]
        if a.limit:
            pp = pp.sample(a.limit, random_state=0)
        jobs = [(pid, r) for pid in pp.index for r in ROTS]
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
                    print(f"{split}: {k + 1}/{len(jobs)} ({el / (k + 1) * 1000:.0f} ms/dosya, kalan "
                          f"{el / (k + 1) * (len(jobs) - k - 1) / 60:.0f} dk)", flush=True)
        df = pd.DataFrame(rows)
        tag = f"_n{a.limit}" if a.limit else ""
        df.to_csv(OUT / f"vol_feats_{a.tag}_{split}{tag}.csv", index=False)
        pos = df.merge(pp[["label"]], left_on="pair_id", right_index=True)
        pos = pos[pos.label == "pos"]
        print(f"{split}: {len(df)} satır, {time.time() - t0:.0f} s; pozitif (tüm dönmeler) tepe hatası medyan "
              f"{pos.err_m.median():.1f} m, ≤ 25 m %{100 * (pos.err_m <= 25).mean():.1f}", flush=True)
        if not a.limit and a.tag == "v1":
            log_run("60_m2_features.py", Path("configs/m2.yaml"), 0, split, "n_rows", len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
