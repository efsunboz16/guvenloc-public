"""UAV-VisLoc okuyucu (Faz 1B).

Doğrulanmış paket yapısı (docs/faz1_veri_notlari.md, örnek paket bölge 03):
    <kök>/satellite_ coordinates_range.csv   # adı boşluklu olabilir
        sütunlar: mapname, LT_lat_map, LT_lon_map, RB_lat_map, RB_lon_map, region
        mapname "03.tif" yazar ama dosya satellite03.tif'tir
    <kök>/NN/NN.csv                          # num, filename, date, lat, lon, height,
                                             #   Omega, Kappa, Phi1, Phi2
    <kök>/NN/drone/*.JPG
    <kök>/NN/satelliteNN.tif                 # EPSG:4326: piksel metrede kare DEĞİL

`lat`, `lon` drone karesinin merkezidir. Phi1/Phi2 yaw'dır (Phi1 daha güvenilir);
işareti ve referansı henüz doğrulanmadı (scripts/12_overlay_check.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from guvenloc.geo import GeoRef

RANGE_GLOB = "satellite*coordinates*range*.csv"


def find_root(root: str | Path) -> Path:
    """Zip tek üst klasörle açıldıysa bölge klasörlerini içeren dizine iner."""
    root = Path(root)
    while root.is_dir() and not any(
        p.is_dir() and re.fullmatch(r"\d{2}", p.name) for p in root.iterdir()
    ):
        subdirs = [p for p in root.iterdir() if p.is_dir()]
        if len(subdirs) != 1:
            break
        root = subdirs[0]
    return root


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_ranges(root: Path) -> dict[str, tuple[float, float, float, float]]:
    """{bölge_id: (lat_top, lon_left, lat_bottom, lon_right)}."""
    files = sorted(root.glob(RANGE_GLOB))
    if not files:
        raise FileNotFoundError(f"{root} altında satellite*coordinates*range*.csv yok")
    df = _read_csv(files[0])
    out = {}
    for _, r in df.iterrows():
        rid = re.sub(r"\D", "", str(r["mapname"])).zfill(2)
        out[rid] = (
            float(r["LT_lat_map"]), float(r["LT_lon_map"]),
            float(r["RB_lat_map"]), float(r["RB_lon_map"]),
        )
    return out


def list_regions(root: str | Path) -> list[str]:
    root = find_root(root)
    return sorted(p.name for p in root.iterdir() if p.is_dir() and re.fullmatch(r"\d{2}", p.name))


@dataclass
class Region:
    region_id: str
    dir: Path
    frames: pd.DataFrame      # bölge CSV'si (num, filename, lat, lon, height, Phi1, ...)
    sat_path: Path
    georef: GeoRef

    def frame_path(self, filename: str) -> Path:
        return self.dir / "drone" / filename

    def yaw(self, row: pd.Series) -> float:
        """Phi1, boşsa Phi2 (README: Phi1 daha güvenilir). İkisi de boşsa NaN."""
        for c in ("Phi1", "Phi2"):
            if c in row and pd.notna(row[c]):
                return float(row[c])
        return float("nan")


def load_region(
    root: str | Path, region_id: str, raster_size: tuple[int, int] | None = None
) -> Region:
    """Bölgeyi yükler. raster_size=(genişlik, yükseklik) yalnızca testte rasterio'suz kullanım için."""
    root = find_root(root)
    rid = str(region_id).zfill(2)
    rdir = root / rid
    csvs = sorted(rdir.glob("*.csv"))
    if len(csvs) != 1:
        raise FileNotFoundError(f"{rdir}: 1 CSV bekleniyordu, {len(csvs)} bulundu")
    frames = _read_csv(csvs[0])
    tifs = sorted(rdir.glob("satellite*.tif"))
    if len(tifs) != 1:
        raise FileNotFoundError(f"{rdir}: 1 uydu tif'i bekleniyordu, {len(tifs)} bulundu")
    rng = read_ranges(root)
    if rid not in rng:
        raise KeyError(f"{rid} koordinat aralığı dosyasında yok: {sorted(rng)}")
    if raster_size is None:
        import rasterio
        with rasterio.open(tifs[0]) as ds:
            raster_size = (ds.width, ds.height)
    lat_t, lon_l, lat_b, lon_r = rng[rid]
    ref = GeoRef(lat_t, lon_l, lat_b, lon_r, raster_size[0], raster_size[1])
    return Region(rid, rdir, frames, tifs[0], ref)


def crop_square_m(
    region: Region, lat: float, lon: float, size_m: float, out_px: int
) -> np.ndarray:
    """(lat, lon) merkezli, kenarı `size_m` metre olan KARE karoyu out_px x out_px'e örnekler.

    Uydu pikseli metrede kare olmadığından (EPSG:4326) pencere eksen başına metreden
    hesaplanır ve rasterio çıktıyı kare ızgaraya yeniden örnekler; sonuçta
    1 piksel = size_m / out_px metre, iki eksende de aynı. Harita dışı kalan yer 0 olur.
    Döndürür: (out_px, out_px, bant) uint8.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    mx, my = region.georef.meters_per_px()
    cx, cy = region.georef.latlon_to_px(lat, lon)
    w_px, h_px = size_m / mx, size_m / my
    win = Window(cx - w_px / 2, cy - h_px / 2, w_px, h_px)
    with rasterio.open(region.sat_path) as ds:
        arr = ds.read(
            window=win, out_shape=(ds.count, out_px, out_px), boundless=True,
            fill_value=0, resampling=Resampling.bilinear,
        )
    return np.moveaxis(arr, 0, -1)
