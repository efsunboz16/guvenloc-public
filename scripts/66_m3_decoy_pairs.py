"""Faz 4, karar F4-5 – gerçek tuzak çiftleri: her drone karesi × başka bölgeden rastgele (easy) karo.

  python scripts/66_m3_decoy_pairs.py         # → data/pairs/decoy_v1.csv (train + calib; test Faz 6'da)

Tuzak karo = komşu karenin (aynı bölge, indeks +1, yoksa −1) easy0 karosunun konumu (bölge, enlem, boylam);
boyut ve ölçek karenin kendi easy0 satırından (tile_size_m, tile_px, scale_s karenin irtifasına bağlı).
Böylece tuzak, karenin kendi negatiflerinin hiçbiriyle aynı değil ve pairs.py'nin easy kuralıyla aynı havuzdan.
Önbellek: python scripts/30_build_cache.py data/UAV_VisLoc_dataset --pairs data/pairs/decoy_v1.csv --cache cache/decoy_v1
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

OUT = Path("data/pairs/decoy_v1.csv")


def main() -> int:
    p = pd.read_csv("data/pairs/pairs_v1.csv", dtype={"region": str, "tile_region": str})
    p = p[p.split.isin(["train", "calib"])]
    e = p[p.pair_id.str.endswith("_easy0")].copy()
    e["fidx"] = e.pair_id.str.split("_").str[1].astype(int)
    loc = {(r.region, r.fidx): (r.tile_region, r.tile_lat, r.tile_lon) for r in e.itertuples()}
    rows = []
    for r in e.itertuples(index=False):
        nb = loc.get((r.region, r.fidx + 1)) or loc.get((r.region, r.fidx - 1))
        if nb is None:
            continue
        d = r._asdict()
        stem = r.pair_id.removesuffix("_easy0")
        d.update(pair_id=f"{stem}_decoy0", tile_path=f"tiles/{stem}_decoy0.jpg", neg_type="easy",
                 tile_region=nb[0], tile_lat=nb[1], tile_lon=nb[2])
        d.pop("fidx")
        rows.append(d)
    out = pd.DataFrame(rows)[[c for c in p.columns]]
    out.to_csv(OUT, index=False)
    print(f"{OUT}: {len(out)} tuzak çifti (" + ", ".join(f"{k} {v}" for k, v in out.split.value_counts().items()) + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
