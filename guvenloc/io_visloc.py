"""UAV-VisLoc okuyucu (Faz 1B).

Doğrulanmış paket yapısı (docs/faz1_veri_notlari.md, örnek paket bölge 03):
    <kök>/satellite_ coordinates_range.csv   # adı boşluklu olabilir
        sütunlar: mapname, LT_lat_map, LT_lon_map, RB_lat_map, RB_lon_map, region
        mapname "03.tif" yazar ama dosya satellite03.tif'tir
    <kök>/NN/NN.csv                          # num, filename, date, lat, lon, height,
                                             #   Omega, Kappa, Phi1, Phi2
    <kök>/NN/drone/*.JPG
    <kök>/NN/satelliteNN.tif                 # EPSG:4326: piksel metrede kare DEĞİL
    <kök>/09/satellite09_RR-CC.tif           # tam pakette 09 dört parça (2x2)

Tam paket bulguları (docs/faz1_veri_notlari.md):
  * Georeferansın kaynağı tif'in kendisidir. 09'da koordinat CSV'si tif sınırlarından
    52 m'ye kadar sapıyor; diğer bölgelerde ikisi birebir aynı.
  * Birden çok tif varsa bölge klasörüne `satelliteNN_mosaic.vrt` (GDAL sanal mozaik)
    yazılır ve bölge tek harita gibi okunur.

`lat`, `lon` drone karesinin merkezidir. Phi1/Phi2 yaw'dır (Phi1 daha güvenilir);
işareti ve referansı henüz doğrulanmadı (scripts/12_overlay_check.py).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

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


def satellite_tifs(rdir: Path) -> list[Path]:
    return sorted(p for p in rdir.glob("satellite*.tif"))


def build_mosaic_vrt(tifs: list[Path], out: Path) -> Path:
    """Aynı CRS ve çözünürlükteki bitişik tif'leri tek bir VRT mozaiğinde birleştirir."""
    import rasterio

    infos = []
    for t in tifs:
        with rasterio.open(t) as ds:
            infos.append((t, ds.bounds, ds.res, ds.width, ds.height, ds.count, ds.dtypes[0], ds.crs))
    res = infos[0][2]
    crs, count, dtype = infos[0][7], infos[0][5], infos[0][6]
    for t, _, r, *_rest, c in infos:
        if c != crs or not np.allclose(r, res, rtol=1e-6):
            raise ValueError(f"{t}: CRS/çözünürlük mozaikteki diğer parçalarla aynı değil")
    left = min(i[1].left for i in infos)
    top = max(i[1].top for i in infos)
    right = max(i[1].right for i in infos)
    bottom = min(i[1].bottom for i in infos)
    W = round((right - left) / res[0])
    H = round((top - bottom) / res[1])
    gdal_type = {"uint8": "Byte", "uint16": "UInt16", "float32": "Float32"}[dtype]
    lines = [
        f'<VRTDataset rasterXSize="{W}" rasterYSize="{H}">',
        f"  <SRS>{escape(crs.to_wkt())}</SRS>",
        f"  <GeoTransform>{left!r}, {res[0]!r}, 0, {top!r}, 0, {-res[1]!r}</GeoTransform>",
    ]
    for b in range(1, count + 1):
        lines.append(f'  <VRTRasterBand dataType="{gdal_type}" band="{b}">')
        for t, bd, _, w, h, *_ in infos:
            xo = round((bd.left - left) / res[0])
            yo = round((top - bd.top) / res[1])
            lines += [
                "    <SimpleSource>",
                f'      <SourceFilename relativeToVRT="1">{escape(t.name)}</SourceFilename>',
                f"      <SourceBand>{b}</SourceBand>",
                f'      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>',
                f'      <DstRect xOff="{xo}" yOff="{yo}" xSize="{w}" ySize="{h}"/>',
                "    </SimpleSource>",
            ]
        lines.append("  </VRTRasterBand>")
    lines.append("</VRTDataset>")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def satellite_source(rdir: Path) -> Path:
    """Bölgenin uydu haritası: tek tif ya da (parçalıysa) VRT mozaiği."""
    tifs = satellite_tifs(rdir)
    if not tifs:
        raise FileNotFoundError(f"{rdir}: uydu tif'i yok")
    if len(tifs) == 1:
        return tifs[0]
    vrt = rdir / f"satellite{rdir.name}_mosaic.vrt"
    if not vrt.exists() or vrt.stat().st_mtime < max(t.stat().st_mtime for t in tifs):
        build_mosaic_vrt(tifs, vrt)
    return vrt


def load_region(
    root: str | Path, region_id: str, raster_size: tuple[int, int] | None = None
) -> Region:
    """Bölgeyi yükler; georeferans uydu raster'ının kendisinden okunur.

    raster_size=(genişlik, yükseklik) verilirse rasterio açılmaz ve georeferans koordinat
    CSV'sinden kurulur (yalnızca test/raster'sız kullanım için).
    """
    root = find_root(root)
    rid = str(region_id).zfill(2)
    rdir = root / rid
    csvs = sorted(rdir.glob("*.csv"))
    if len(csvs) != 1:
        raise FileNotFoundError(f"{rdir}: 1 CSV bekleniyordu, {len(csvs)} bulundu")
    frames = _read_csv(csvs[0])
    if raster_size is not None:
        tifs = satellite_tifs(rdir)
        if not tifs:
            raise FileNotFoundError(f"{rdir}: uydu tif'i yok")
        rng = read_ranges(root)
        if rid not in rng:
            raise KeyError(f"{rid} koordinat aralığı dosyasında yok: {sorted(rng)}")
        lat_t, lon_l, lat_b, lon_r = rng[rid]
        return Region(rid, rdir, frames, tifs[0],
                      GeoRef(lat_t, lon_l, lat_b, lon_r, raster_size[0], raster_size[1]))

    import rasterio

    sat = satellite_source(rdir)
    with rasterio.open(sat) as ds:
        if ds.crs is None or ds.crs.to_epsg() != 4326:
            raise ValueError(f"{sat}: EPSG:4326 bekleniyordu, {ds.crs} bulundu")
        b = ds.bounds
        ref = GeoRef(b.top, b.left, b.bottom, b.right, ds.width, ds.height)
    return Region(rid, rdir, frames, sat, ref)


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
