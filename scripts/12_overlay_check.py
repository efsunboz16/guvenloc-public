"""Faz 1B sağlama: drone karesi ↔ uydu karosu bindirmesi ve yön (Phi) işareti.

Her seçilen kare için bir JPG üretir (results/overlay/<bölge>_<num>.jpg):
  1. drone karesi
  2. uydu karosu, kuzey yukarı (kare merkezi = drone merkezi)
  3-6. karo, drone yönü yukarı gelecek biçimde döndürülmüş, dört hipotezle:
       +Phi, -Phi, +Phi+180, -Phi+180
Drone karesinin görüntüsüne en çok benzeyen sütun yön sütununun doğru yorumudur
(saat yönü / saat yönünün tersi, 0° = kuzey / güney). Kararı ekip gözle verir ve
guvenloc/geo.py ile guvenloc/io_visloc.py docstring'ine yazılır.

Kullanım:
  python scripts/12_overlay_check.py data/uav_visloc_sample --regions 03 --per-region 3 \
      --footprint-m 600
--footprint-m: karonun kenarı (m). Drone ayak izine yakın seçin; EXIF olmadığından ölçek
bilinmez, birkaç değer deneyin (ör. 400, 600, 900).
Beklenen: toplam 20-30 kare, her bölgeden 2-3.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc.io_visloc import crop_square_m, list_regions, load_region


def rotate_ccw(img: np.ndarray, deg: float) -> np.ndarray:
    """Görüntüyü saat yönünün tersine deg derece döndürür (merkez etrafında)."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--regions", nargs="*", help="örn. 03 04; verilmezse hepsi")
    ap.add_argument("--per-region", type=int, default=3)
    ap.add_argument("--footprint-m", type=float, default=600.0)
    ap.add_argument("--footprint-k", type=float, default=None,
                    help="verilirse karo kenarı = k × irtifa (m); irtifası farklı bölgeler için")
    ap.add_argument("--tile-px", type=int, default=384)
    ap.add_argument("--out", type=Path, default=Path("results/overlay"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    regions = args.regions or list_regions(args.root)
    n_written = 0
    for rid in regions:
        reg = load_region(args.root, rid)
        idx = np.linspace(0, len(reg.frames) - 1, args.per_region).round().astype(int)
        for i in sorted(set(idx)):
            row = reg.frames.iloc[int(i)]
            phi = reg.yaw(row)
            if np.isnan(phi):
                print(f"{rid} {row['filename']}: Phi1/Phi2 boş, yalnız kuzey-yukarı çiziliyor")
            fp = args.footprint_m if args.footprint_k is None else args.footprint_k * float(row["height"])
            tile = crop_square_m(reg, float(row["lat"]), float(row["lon"]), fp, args.tile_px)
            drone = cv2.imread(str(reg.frame_path(str(row["filename"]))))
            panels = [("drone", None if drone is None else cv2.cvtColor(drone, cv2.COLOR_BGR2RGB)),
                      ("uydu, kuzey yukarı", tile)]
            if not np.isnan(phi):
                for label, a in [("+Phi", phi), ("-Phi", -phi),
                                 ("+Phi+180", phi + 180), ("-Phi+180", -phi + 180)]:
                    panels.append((f"{label} ({a:.0f}°)", rotate_ccw(tile, a)))
            fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
            for ax, (title, im) in zip(np.atleast_1d(axes), panels, strict=False):
                if im is not None:
                    ax.imshow(im)
                ax.set_title(title, fontsize=9)
                ax.axis("off")
            fig.suptitle(f"{rid} #{row['num']} {row['filename']}  yaw={phi:.1f}  "
                         f"h={row.get('height', float('nan'))}  footprint={fp:.0f} m",
                         fontsize=9)
            fig.tight_layout()
            path = args.out / f"{rid}_{int(row['num']):04d}.jpg"
            fig.savefig(path, dpi=110)
            plt.close(fig)
            n_written += 1
            print("yazıldı:", path)
    print(f"toplam {n_written} bindirme -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
