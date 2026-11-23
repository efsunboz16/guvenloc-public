"""Faz 1E – RoMa v2 önbelleği: her çift × 4 yön (+ ardışık drone kareleri).

Kullanım:
  python scripts/30_build_cache.py data/UAV_VisLoc_dataset                 # hepsi
  python scripts/30_build_cache.py data/UAV_VisLoc_dataset --split calib   # tek küme
  python scripts/30_build_cache.py data/UAV_VisLoc_dataset --limit 20      # hız ölçümü
  python scripts/30_build_cache.py data/UAV_VisLoc_dataset --split train --per-region 60
  python scripts/30_build_cache.py data/UAV_VisLoc_dataset --seq           # ardışık kareler
  python scripts/30_build_cache.py data/UAV_VisLoc_dataset --finalize      # yalnızca manifest

Yeniden başlatılabilir: var olan .npz atlanır. Manifest (sha256 dahil) yazım sırasında
MANIFEST.partial.csv'ye eklenir, sonda (veya --finalize ile) MANIFEST.csv üretilir.
Ayar ve n configs/base.yaml (romav2.setting_cache, n_matches, rotations) içinden okunur.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc.cache import append_manifest, finalize_manifest, pair_file, seq_file
from guvenloc.io_visloc import crop_square_m, find_root, load_region
from guvenloc.matcher import Matcher
from guvenloc.metrics import log_run


def read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--pairs", type=Path, default=Path("data/pairs/pairs_v1.csv"))
    ap.add_argument("--cache", type=Path, default=None, help="varsayılan: config paths.cache")
    ap.add_argument("--setting", default=None, help="varsayılan: romav2.setting_cache")
    ap.add_argument("--split", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="ilk N drone karesi")
    ap.add_argument("--per-region", type=int, default=None,
                    help="bölge başına eşit aralıklı N drone karesi (ör. eşik seçimi alt kümesi)")
    ap.add_argument("--seq", action="store_true", help="ardışık drone kareleri (Modül 4)")
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rc = cfg["romav2"]
    cache = args.cache or Path(cfg["paths"]["cache"])
    cache.mkdir(parents=True, exist_ok=True)
    partial = cache / "MANIFEST.partial.csv"
    if args.finalize:
        print("manifest:", finalize_manifest(cache))
        return 0

    setting = args.setting or rc["setting_cache"]
    n, rots = int(rc["n_matches"]), [int(r) for r in rc["rotations"]]
    root = find_root(args.root)
    df = pd.read_csv(args.pairs, dtype={"region": str, "tile_region": str})
    if args.split:
        df = df[df.split.isin(args.split)]
    if args.per_region:
        keep = []
        for _, g in df.groupby("region"):
            paths = sorted(g.drone_path.unique())
            idx = np.unique(np.linspace(0, len(paths) - 1, args.per_region).round().astype(int))
            keep += [paths[i] for i in idx]
        df = df[df.drone_path.isin(keep)]
    regions: dict = {}

    def region(rid):
        if rid not in regions:
            regions[rid] = load_region(root, rid)
        return regions[rid]

    matcher = Matcher(setting=setting, sigma_scale=float(rc.get("sigma_scale", 1.0)),
                      seed=int(cfg["seed"]))
    t0, done, skipped = time.perf_counter(), 0, 0

    if args.seq:
        frames = df[df.neg_type == "pos"].sort_values(["region", "drone_path"])
        jobs = []
        for rid, g in frames.groupby("region"):
            nums = g.pair_id.str.split("_").str[1].astype(int).tolist()
            paths = g.drone_path.tolist()
            for (a, pa), (b, pb) in zip(zip(nums, paths), zip(nums[1:], paths[1:]), strict=False):
                jobs.append((rid, a, b, pa, pb))
        jobs = jobs[: args.limit] if args.limit else jobs
        for rid, a, b, pa, pb in jobs:
            out = seq_file(cache, rid, a, b)
            if out.exists():
                skipped += 1
                continue
            ms = matcher.match(read_rgb(root / pa), read_rgb(root / pb), 0, n)
            ms.meta.update(kind="seq", region=rid, t0=a, t1=b)
            ms.save_npz(out)
            append_manifest(partial, out, "seq", f"{rid}_{a:04d}_{b:04d}", 0, ms)
            done += 1
            if done % 50 == 0:
                print(f"  seq {done}/{len(jobs)} | {(time.perf_counter() - t0) / done:.2f} s/çift",
                      flush=True)
    else:
        groups = list(df.groupby("drone_path", sort=True))
        groups = groups[: args.limit] if args.limit else groups

        def prepare(item):
            dpath, g = item
            todo = [(r, rot) for r in g.itertuples() for rot in rots
                    if not pair_file(cache, r.pair_id, rot).exists()]
            if not todo:
                return dpath, None, []
            tiles = {r.pair_id: np.ascontiguousarray(crop_square_m(
                region(r.tile_region), r.tile_lat, r.tile_lon, r.tile_size_m, int(r.tile_px)))
                for r in {t[0] for t in todo}}
            return dpath, read_rgb(root / dpath), [(r, rot, tiles[r.pair_id]) for r, rot in todo]

        # G/Ç (JPEG çözme, raster okuma) bir sonraki kare için arka planda hazırlanır
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut = pool.submit(prepare, groups[0]) if groups else None
            for i in range(len(groups)):
                _dpath, drone, todo = fut.result()
                if i + 1 < len(groups):
                    fut = pool.submit(prepare, groups[i + 1])
                if drone is None:
                    skipped += 1
                    continue
                for r, rot, tile in todo:
                    ms = matcher.match(drone, tile, rot, n)
                    ms.meta.update(kind="pair", pair_id=r.pair_id, label=r.label,
                                   neg_type=r.neg_type, split=r.split)
                    out = pair_file(cache, r.pair_id, rot)
                    ms.save_npz(out)
                    append_manifest(partial, out, "pair", r.pair_id, rot, ms)
                    done += 1
                if (i + 1) % 25 == 0:
                    el = time.perf_counter() - t0
                    print(f"  kare {i + 1}/{len(groups)} | {done} dosya | "
                          f"{el / max(done, 1):.2f} s/dosya | kalan ≈ "
                          f"{el / (i + 1) * (len(groups) - i - 1) / 3600:.1f} sa", flush=True)

    el = time.perf_counter() - t0
    man = finalize_manifest(cache)
    print(f"bitti: {done} yeni, {skipped} atlandı, {el:.0f} s "
          f"({el / max(done, 1):.2f} s/dosya) | manifest {man}")
    if done:
        log_run("30_build_cache.py", args.config, int(cfg["seed"]),
                ",".join(args.split or ["all"]), "sec_per_file", round(el / done, 3),
                notes=f"setting={setting} seq={args.seq} n={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
