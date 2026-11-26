"""Faz 3D – öğrenilmiş inceltici girdisi: v0 hacim tepesi çevresinde kırpıntılar → data/m2/crops_<split>.npy.

  python scripts/62_m2_crops.py --splits calib train     # test Faz 6'ya kadar kapalı (F3-4)

Her (çift, dönme): configs/m2.yaml ile ince hacim (13 θ dilimi, kaba tepe t0 ± 6°) + ağırlıksız yoğunluk hacmi;
tepe hücresi merkezli 2 × 13 × 48 × 48 (±48 m) float16 kırpıntı (kanal 0: c ağırlıklı oy, kanal 1: yoğunluk).
Satır bilgisi (crops_<split>.csv): pair_id, rot, t0, tepe hücresi, cell_px, v0 metrikleri; pozitiflerde GT'nin
kırpıntı koordinatı (gx, gy hücre; gdt = GT yönü − t0, derece).
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc import m2_volume as V

ROOT = Path(__file__).resolve().parents[1]
F60 = __import__("importlib").util.spec_from_file_location("f60", ROOT / "scripts/60_m2_features.py")
f60 = __import__("importlib").util.module_from_spec(F60)
F60.loader.exec_module(f60)
ROTS, HALF, OUT = (0, 90, 180, 270), 24, Path("data/m2")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["calib", "train"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if "test" in a.splits:
        sys.exit("test bölmesi Faz 6'ya kadar kapalı (F3-4)")
    OUT.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    mcfg = yaml.safe_load(Path("configs/m2.yaml").read_text(encoding="utf-8"))
    pairs = pd.read_csv("data/pairs/pairs_v1.csv", dtype={"region": str, "tile_region": str}).set_index("pair_id")
    for split in a.splits:
        pp = pairs[pairs.split == split]
        if a.limit:
            pp = pp.sample(a.limit, random_state=0)
        jobs = [(pid, r) for pid in pp.index for r in ROTS]
        T = int(2 * mcfg["fine_half_deg"] / mcfg["fine_step_deg"]) + 1
        tag = f"_n{a.limit}" if a.limit else ""
        arr = np.lib.format.open_memmap(OUT / f"crops_{split}{tag}.npy", mode="w+", dtype=np.float16,
                                        shape=(len(jobs), 2, T, 2 * HALF, 2 * HALF))
        rows, t0 = [], time.time()
        with ThreadPoolExecutor(6) as ex:
            for k, (pid, rot, z, meta) in enumerate(ex.map(f60.load, jobs)):
                m = pp.loc[pid]
                cfg = V.VolumeCfg(**mcfg, sigma_scale=float(base["romav2"]["sigma_scale"]) / float(meta["sigma_scale"]))
                mpp = float(m.tile_size_m) / float(m.tile_px)
                vol = V.build_volume(z["u"], z["v"], z["c"], z["Sigma_B"], z["Sigma_A"], float(m.scale_s),
                                     meta["drone_wh"], int(m.tile_px), mpp, cfg, density=True)
                r = V.metrics(vol)
                cp = vol["cell_px"]
                if np.isfinite(r["x"]):
                    cx, cy = round((r["x"] + 0.5) / cp - 0.5), round((r["y"] + 0.5) / cp - 0.5)
                else:
                    cx = cy = vol["n"] // 2
                arr[k, 0] = V.crop(vol["vol"], cy, cx, HALF).cpu().numpy().astype(np.float16)
                arr[k, 1] = V.crop(vol["dens"], cy, cx, HALF).cpu().numpy().astype(np.float16)
                th0 = float(vol["thetas"][T // 2])
                row = {"pair_id": pid, "rot": rot, "t0": th0, "cx": cx, "cy": cy, "cell_px": cp, "mpp": mpp, **r}
                row["err_m"] = float(np.hypot(r["x"] - m.gt_x, r["y"] - m.gt_y) * mpp) if m.label == "pos" else np.nan
                if m.label == "pos":
                    row["gx"] = (m.gt_x + 0.5) / cp - 0.5 - (cx - HALF)
                    row["gy"] = (m.gt_y + 0.5) / cp - 0.5 - (cy - HALF)
                    row["gdt"] = (m.gt_heading - th0 + 180) % 360 - 180
                rows.append(row)
                if (k + 1) % 2000 == 0:
                    el = time.time() - t0
                    print(f"{split}: {k + 1}/{len(jobs)} ({el / (k + 1) * 1000:.0f} ms/dosya, kalan "
                          f"{el / (k + 1) * (len(jobs) - k - 1) / 60:.0f} dk)", flush=True)
        arr.flush()
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"crops_{split}{tag}.csv", index=False)
        inside = df.dropna(subset=["gx"])
        ins = ((inside.gx >= 0) & (inside.gx < 2 * HALF) & (inside.gy >= 0) & (inside.gy < 2 * HALF)).mean()
        print(f"{split}: {len(df)} satır, {time.time() - t0:.0f} s; pozitif tepe hatası medyan "
              f"{df.err_m.median():.1f} m, ≤ 25 m %{100 * (df.err_m <= 25).mean():.1f}; GT kırpıntı içinde %{100 * ins:.1f}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
