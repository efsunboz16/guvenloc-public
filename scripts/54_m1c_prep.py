"""Faz 2C (öneri C fizibilitesi) – her M1 karesi için kuzey-yukarı uydu karosu → data/m1/relsat_321.npy.

  python scripts/54_m1c_prep.py
  python scripts/54_m1c_prep.py --gercek       # sistemdeki gerçek karo → data/m1/relsat_real_513.npy
  python scripts/54_m1c_prep.py --gercek --ek 3   # eğitim karesi başına 1 + 3 karo → relsat_real4_513.npy

--gercek: val/eval için pairs_v1 pozitif karosunun kendisi (tile_lat/lon, tile_size_m = ayak izi köşegeni
+ 400 m, merkez GT'den ≤ 200 m); eğitim için aynı karo + aynı kuralla 1 rastgele karo daha (karesi başına 2).
Karo 513 px (drone girdisi 257 px / merkez karesi → çözünürlük oranı ≈ 1,3–1,85).

Karo kenarı = 1,5 × drone merkez karesinin ayak izi (0,98 × AGL × 2652/3976 m); merkez GT'den diskte düzgün
kaydırılmış, yarıçap ≤ 0,25 × ayak izi. Kolay kurulum: sistemdeki gerçek karo (ayak izi köşegeni + 400 m,
≈ 3×) daha büyük → fizibilite olumluysa karo büyütülerek tekrarlanmalı (docs/faz2_m1.md 2C-C).
Eğitim (m1_role train) karesi başına 3 karo, val/eval karesi başına 1 karo (sabit tohum). Her bölgenin
kendi haritası; val/eval haritası yalnız değerlendirmede kullanılır (çıkarımda karo zaten verilir).
Çıktı satırları: relsat_321.csv (lab_idx = labels_v2 satırı, m1_role, region, side_m, off_m).
"""
from __future__ import annotations

import argparse
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
from guvenloc.pairs import offset_latlon, tile_size_for

DATA = Path("data/UAV_VisLoc_dataset")
OUT = Path("data/m1/relsat_321.npy")
SIZE, SIDE_X, OFF_X, SEED = 321, 1.5, 0.25, 1
PER = {"train": 3, "val": 1, "eval": 1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gercek", action="store_true")
    ap.add_argument("--ek", type=int, default=1, help="--gercek: eğitim karesi başına ek rastgele karo")
    a = ap.parse_args()
    pcfg = yaml.safe_load(Path("configs/pairs.yaml").read_text(encoding="utf-8"))
    cam = pcfg["camera"]
    lab = pd.read_csv("data/m1/labels_v2.csv", dtype={"region": str})
    pr = pd.read_csv("data/pairs/pairs_v1.csv", dtype={"region": str}).set_index("pair_id")
    lab = lab.join(pr[["drone_lat", "drone_lon", "agl_m", "tile_lat", "tile_lon", "tile_size_m"]], on="pair_id")
    regions = {r: load_region(DATA, r) for r in sorted(lab.region.unique())}
    rng = np.random.default_rng(SEED)
    jobs = []
    size, out = SIZE, OUT
    if a.gercek:
        size = 513
        out = Path("data/m1/relsat_real_513.npy" if a.ek == 1 else f"data/m1/relsat_real{a.ek + 1}_513.npy")
        off_max = float(pcfg["tile_offset_m"])
        for i, row in enumerate(lab.itertuples()):
            jobs.append((i, row.m1_role, row.region, row.tile_lat, row.tile_lon, row.tile_size_m, np.nan))
            for _ in range(a.ek if row.m1_role == "train" else 0):
                r, ang = off_max * math.sqrt(rng.uniform()), rng.uniform(0, 2 * math.pi)
                lat, lon = offset_latlon(row.drone_lat, row.drone_lon, r * math.cos(ang), r * math.sin(ang))
                jobs.append((i, row.m1_role, row.region, lat, lon, tile_size_for(row.agl_m, pcfg), r))
    for i, row in enumerate([] if a.gercek else lab.itertuples()):
        fp = cam["footprint_over_agl"] * row.agl_m * cam["image_height_px"] / cam["image_width_px"]
        for _ in range(PER[row.m1_role]):
            r, a = OFF_X * fp * math.sqrt(rng.uniform()), rng.uniform(0, 2 * math.pi)
            lat, lon = offset_latlon(row.drone_lat, row.drone_lon, r * math.cos(a), r * math.sin(a))
            jobs.append((i, row.m1_role, row.region, lat, lon, SIDE_X * fp, r))
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=(len(jobs), size, size, 3))

        def put(k):
            j = jobs[k]
            arr[k] = crop_square_m(regions[j[2]], j[3], j[4], j[5], size)[..., :3]

        list(ex.map(put, range(len(jobs))))
    arr.flush()
    pd.DataFrame(jobs, columns=["lab_idx", "m1_role", "region", "lat", "lon", "side_m", "off_m"]).to_csv(
        out.with_suffix(".csv"), index=False)
    print(f"{out}: {arr.shape} {arr.nbytes / 1e9:.2f} GB, {time.time() - t0:.0f} s; "
          f"rol başına {pd.Series([j[1] for j in jobs]).value_counts().to_dict()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
