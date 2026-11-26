"""Faz 2 (devam), karar F2-11 – M1-C tohum birleşimi / test zamanı döndürme: varyant seçimi YALNIZ val 04'te.

  python scripts/58_m1c_ens.py

Ölçüt H1 (i) ile aynı: pozitif çiftlerde, kare olduğu gibi (0°), medyan yön hatası. Seçim kuralı (sonuçtan önce
yazıldı): val 04 medyanı en küçük varyant; en iyiden ≤ 0,1° farklı daha ucuz (daha az ileri geçiş) varyant
varsa o seçilir. Calib'e dokunmaz → results/faz2/m1/ens_sel.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from guvenloc import m1_canon as M

spec = importlib.util.spec_from_file_location("m1c56", ROOT / "scripts/56_m1c_roma.py")
R56 = importlib.util.module_from_spec(spec)
sys.modules["m1c56"] = R56
spec.loader.exec_module(R56)

SEEDS = ["D2_olcek_rot90", "D2_olcek_rot90_s1", "D2_olcek_rot90_s2"]
ENS = "+".join(SEEDS)
VARIANTS = {"D2_olcek_rot90_s2": 1, "D2_olcek_rot90_s2@tta": 4, ENS: 3, ENS + "@tta": 12}   # ad → ileri geçiş


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pairs_all = pd.read_csv("data/pairs/pairs_v1.csv", dtype={"region": str, "tile_region": str})
    pairs = pairs_all[pairs_all.region == "04"].sort_values("pair_id").reset_index(drop=True)
    tiles = R56.tiles_for(pairs, Path("data/m1/val04_pairs_513.npy"))
    pos = (pairs.label == "pos").to_numpy()
    gt = torch.tensor(pairs.gt_heading.to_numpy()[pos] % 360, dtype=torch.float32)
    res = {}
    for v, cost in VARIANTS.items():
        _, phi = R56.m1_rot_probs(v, pairs, tiles)
        e = M.angular_error_deg(torch.tensor(phi[pos]), gt).numpy()
        res[v] = {"median": float(np.median(e)), "acc45": float((e <= 45).mean()), "passes": cost}
        print(f"val 04 {v}: medyan {res[v]['median']:.2f}°, ≤45° %{100 * res[v]['acc45']:.1f}, {cost} geçiş")
    best = min(res, key=lambda k: res[k]["median"])
    sel = min((k for k in res if res[k]["median"] <= res[best]["median"] + 0.1), key=lambda k: res[k]["passes"])
    out = {"region": "04", "n_pos": int(pos.sum()), "variants": res, "best": best, "selected": sel}
    Path("results/faz2/m1/ens_sel.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"seçilen: {sel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
