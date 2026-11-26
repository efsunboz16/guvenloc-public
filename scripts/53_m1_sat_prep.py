"""Faz 2C (öneri B) – M1 için kuzey-yukarı uydu kırpıntıları → data/m1/sat_321.npy (+ sat_321.csv).

  python scripts/53_m1_sat_prep.py              # eğitim karesi başına 3 kırpıntı

Yalnız m1_role == train bölgeleri (01, 08, 09, 10): doğrulama 04 ve calib 05/11 haritası kullanılmaz.
Her eğitim karesinin GT konumu çevresinde (diskte düzgün, yarıçap ≤ 300 m) kırpıntı alınır; kenar = drone
merkez karesinin ayak izi (0,98 × AGL × 2652/3976 m) × U(0,8; 1,25). Uydu karosu kuzey-yukarı → φ = 0.
Harita dışına %10'dan fazla taşan kırpıntı yeniden örneklenir.
"""
from __future__ import annotations

import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc.io_visloc import crop_square_m, load_region
from guvenloc.pairs import offset_latlon, tile_in_map_fraction

DATA = Path("data/UAV_VisLoc_dataset")
LABELS = Path("data/m1/labels_v2.csv")
PAIRS = Path("data/pairs/pairs_v1.csv")
OUT = Path("data/m1/sat_321.npy")
SIZE = 321
PER_FRAME = 3
MAX_OFF_M = 300.0
SEED = 0


def main() -> int:
    cam = yaml.safe_load(Path("configs/pairs.yaml").read_text(encoding="utf-8"))["camera"]
    lab = pd.read_csv(LABELS, dtype={"region": str})
    lab = lab[lab.m1_role == "train"]
    pr = pd.read_csv(PAIRS, dtype={"region": str}).set_index("pair_id")
    lab = lab.join(pr[["drone_lat", "drone_lon", "agl_m"]], on="pair_id")
    regions = {r: load_region(DATA, r) for r in sorted(lab.region.unique())}
    rng = np.random.default_rng(SEED)
    jobs = []
    for row in lab.itertuples():
        side = cam["footprint_over_agl"] * row.agl_m * cam["image_height_px"] / cam["image_width_px"]
        for _ in range(PER_FRAME):
            for _try in range(50):
                r, a = MAX_OFF_M * math.sqrt(rng.uniform()), rng.uniform(0, 2 * math.pi)
                lat, lon = offset_latlon(row.drone_lat, row.drone_lon, r * math.cos(a), r * math.sin(a))
                s = side * math.exp(rng.uniform(math.log(0.8), math.log(1.25)))
                if tile_in_map_fraction(regions[row.region].georef, lat, lon, s) >= 0.9:
                    break
            jobs.append((row.region, lat, lon, s, row.pair_id))
    t0 = time.time()

    def crop(j):
        im = crop_square_m(regions[j[0]], j[1], j[2], j[3], SIZE)
        return im[..., :3]

    with ThreadPoolExecutor(8) as ex:
        arr = np.stack(list(ex.map(crop, jobs)))
    np.save(OUT, arr)
    pd.DataFrame(jobs, columns=["region", "lat", "lon", "side_m", "src_pair_id"]).to_csv(
        OUT.with_suffix(".csv"), index=False)
    print(f"{OUT}: {arr.shape} {arr.nbytes / 1e9:.2f} GB, {time.time() - t0:.0f} s; "
          f"bölge başına {pd.Series([j[0] for j in jobs]).value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
