"""Faz 1C – çift listesi: data/pairs/pairs_v1.csv (INTERFACES.md §2 + ek alanlar).

Kullanım:
  python scripts/14_make_pairs.py data/UAV_VisLoc_dataset
  python scripts/14_make_pairs.py data/UAV_VisLoc_dataset --check 50   # + görsel kontrol
  python scripts/14_make_pairs.py data/UAV_VisLoc_dataset --cut-tiles  # karo görüntülerini kes

--check N: rastgele N çift için results/faz1/pairs_check/<pair_id>.jpg — solda drone karesi,
sağda karo; pozitiflerde GT ayak izi (yönüyle) karo üzerine çizilir.
Bitti sayılır (MIMARI 1C): aynı seed ile yeniden üretim birebir aynı (betik sha256 yazar).
"""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc.io_visloc import crop_square_m, find_root, list_regions, load_region
from guvenloc.metrics import log_run
from guvenloc.pairs import RegionInfo, load_split, make_pairs


def footprint_corners(row, img_w: int, img_h: int) -> np.ndarray:
    """GT drone ayak izinin karo pikselindeki 4 köşesi (görüntü üst kenarı kırmızı çizilir)."""
    k = row.drone_m_per_px / (row.tile_size_m / row.tile_px)
    c = np.array([[-img_w / 2, -img_h / 2], [img_w / 2, -img_h / 2],
                  [img_w / 2, img_h / 2], [-img_w / 2, img_h / 2]]) * k
    t = math.radians(row.gt_heading)  # saat yönü (y aşağı koordinatta pozitif açı)
    R = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return c @ R.T + [row.gt_x, row.gt_y]


def draw_check(row, root: Path, regions: dict, out: Path) -> None:
    drone = cv2.imread(str(root / row.drone_path))
    tile = crop_square_m(regions[row.tile_region], row.tile_lat, row.tile_lon,
                         row.tile_size_m, row.tile_px)
    tile = cv2.cvtColor(np.ascontiguousarray(tile), cv2.COLOR_RGB2BGR)
    if row.label == "pos" and np.isfinite(row.gt_heading):
        pts = footprint_corners(row, drone.shape[1], drone.shape[0]).round().astype(np.int32)
        cv2.polylines(tile, [pts], True, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.line(tile, tuple(pts[0]), tuple(pts[1]), (0, 0, 255), 6, cv2.LINE_AA)  # üst kenar
        cv2.circle(tile, (int(row.gt_x), int(row.gt_y)), 8, (0, 0, 255), -1)
    h = tile.shape[0]
    d = cv2.resize(drone, (int(drone.shape[1] * h / drone.shape[0]), h))
    canvas = np.hstack([d, tile])
    txt = (f"{row.pair_id} {row.split} {row.neg_type} tile@{row.tile_region} "
           f"yaw={row.gt_heading:.0f} agl={row.agl_m:.0f} s={row.scale_s:.3f}")
    cv2.putText(canvas, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.imwrite(str(out / f"{row.pair_id}.jpg"), cv2.resize(canvas, None, fx=0.5, fy=0.5))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--split", type=Path, default=Path("configs/split.yaml"))
    ap.add_argument("--config", type=Path, default=Path("configs/pairs.yaml"))
    ap.add_argument("--out", type=Path, default=Path("data/pairs"))
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--cut-tiles", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    split = load_split(args.split)
    root = find_root(args.root)
    regs = {r: load_region(root, r) for r in list_regions(root)
            if r in split.region_split}
    infos = {r: RegionInfo(r, g.frames, g.georef) for r, g in regs.items()}
    df = make_pairs(infos, split, cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / f"{cfg['version']}.csv"
    df.to_csv(csv_path, index=False, float_format="%.8f", lineterminator="\n")
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    print(f"yazıldı: {csv_path} ({len(df)} satır) sha256={digest[:16]}")
    print(df.groupby(["split", "neg_type"]).size().unstack(fill_value=0).to_string())
    print("bölge başına zor negatif mesafesi (medyan m) ve eksik zor negatif:")
    for rid, g in df.groupby("region"):
        n_pos = (g.neg_type == "pos").sum()
        hard = g[g.neg_type == "hard"]
        print(f"  {rid}: pos {n_pos:5d} | hard {len(hard):5d} (eksik {n_pos - len(hard)}), "
              f"medyan {hard.neg_dist_m.median():6.0f} m | pozitif karo harita içi "
              f"medyan {g[g.neg_type == 'pos'].tile_in_map.median():.2f}")

    for (sp, nt), g in df.groupby(["split", "neg_type"]):
        log_run("14_make_pairs.py", args.config, int(cfg["seed"]), sp, f"n_pairs_{nt}", len(g),
                notes=f"{cfg['version']} sha256={digest[:16]}")

    if args.check:
        out = Path("results/faz1/pairs_check")
        out.mkdir(parents=True, exist_ok=True)
        for _, row in df.sample(args.check, random_state=int(cfg["seed"])).iterrows():
            draw_check(row, root, regs, out)
        print(f"görsel kontrol: {out} ({args.check} çift)")

    if args.cut_tiles:
        for i, row in enumerate(df.itertuples(), 1):
            p = args.out / row.tile_path
            if p.exists():
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            t = crop_square_m(regs[row.tile_region], row.tile_lat, row.tile_lon,
                              row.tile_size_m, row.tile_px)
            cv2.imwrite(str(p), cv2.cvtColor(np.ascontiguousarray(t), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            if i % 1000 == 0:
                print(f"  karo {i}/{len(df)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
