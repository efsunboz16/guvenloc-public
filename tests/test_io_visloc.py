import numpy as np
import pytest

pytest.importorskip("rasterio")
pytest.importorskip("pandas")

import rasterio
from rasterio.transform import from_bounds

from guvenloc.io_visloc import crop_square_m, list_regions, load_region, read_ranges

LT_LAT, LT_LON, RB_LAT, RB_LON = 32.40, 120.00, 32.30, 120.10
W, H = 400, 300


def make_pkg(root):
    (root / "03" / "drone").mkdir(parents=True)
    # Gerçek paketteki gibi boşluklu dosya adı ve "03.tif" mapname
    (root / "satellite_ coordinates_range.csv").write_text(
        "mapname,LT_lat_map,LT_lon_map,RB_lat_map,RB_lon_map,region\n"
        f"03.tif,{LT_LAT},{LT_LON},{RB_LAT},{RB_LON},Taizhou\n"
    )
    (root / "03" / "03.csv").write_text(
        "num,filename,date,lat,lon,height,Omega,Kappa,Phi1,Phi2\n"
        "1,03_0001.JPG,2023-01-01,32.35,120.05,466,0.1,0.2,45.0,40.0\n"
        "2,03_0002.JPG,2023-01-01,32.36,120.06,466,0.1,0.2,,12.0\n"
    )
    img = np.zeros((3, H, W), dtype="uint8")
    img[0, :, W // 2:] = 255          # sağ yarı kırmızı
    img[1, H // 2:, :] = 255          # alt yarı yeşil
    with rasterio.open(
        root / "03" / "satellite03.tif", "w", driver="GTiff", height=H, width=W, count=3,
        dtype="uint8", crs="EPSG:4326",
        transform=from_bounds(LT_LON, RB_LAT, RB_LON, LT_LAT, W, H),
    ) as ds:
        ds.write(img)


def test_load_region(tmp_path):
    make_pkg(tmp_path)
    assert list_regions(tmp_path) == ["03"]
    assert "03" in read_ranges(tmp_path)
    reg = load_region(tmp_path, "3")
    assert reg.georef.width_px == W and reg.georef.height_px == H
    assert len(reg.frames) == 2
    assert reg.yaw(reg.frames.iloc[0]) == 45.0
    assert reg.yaw(reg.frames.iloc[1]) == 12.0     # Phi1 boş -> Phi2
    assert reg.frame_path("03_0001.JPG").name == "03_0001.JPG"


def test_crop_is_square_and_oriented(tmp_path):
    make_pkg(tmp_path)
    reg = load_region(tmp_path, "03")
    mx, my = reg.georef.meters_per_px()
    assert abs(mx - my) / my > 0.05               # piksel kare değil
    crop = crop_square_m(reg, 32.35, 120.05, size_m=2000, out_px=64)
    assert crop.shape == (64, 64, 3)
    # Merkez noktası sağ/alt sınırında: sol-üst karede kırmızı yok, sağ-alt karede var
    assert crop[10, 5, 0] == 0 and crop[10, 60, 0] == 255
    assert crop[5, 10, 1] == 0 and crop[60, 10, 1] == 255


def test_crop_outside_map_is_zero(tmp_path):
    make_pkg(tmp_path)
    reg = load_region(tmp_path, "03")
    crop = crop_square_m(reg, 33.5, 120.05, size_m=500, out_px=16)
    assert crop.max() == 0


def test_mosaic_vrt_tiles_match_single_raster(tmp_path):
    """Parçalı harita (tam pakette 09): 2x2 parçanın VRT'si tek tif ile aynı pikselleri vermeli."""
    rdir = tmp_path / "09"
    (rdir / "drone").mkdir(parents=True)
    rng = np.random.default_rng(0)
    full = rng.integers(0, 255, (3, H, W), dtype="uint8")
    tf = from_bounds(LT_LON, RB_LAT, RB_LON, LT_LAT, W, H)
    xs, ys = (0, 250, W), (0, 100, H)       # eşit olmayan parçalar
    for i in range(2):
        for j in range(2):
            y0, y1, x0, x1 = ys[i], ys[i + 1], xs[j], xs[j + 1]
            with rasterio.open(
                rdir / f"satellite09_{i + 1:02d}-{j + 1:02d}.tif", "w", driver="GTiff",
                height=y1 - y0, width=x1 - x0, count=3, dtype="uint8", crs="EPSG:4326",
                transform=rasterio.windows.transform(
                    rasterio.windows.Window(x0, y0, x1 - x0, y1 - y0), tf),
            ) as ds:
                ds.write(full[:, y0:y1, x0:x1])
    (rdir / "09.csv").write_text(
        "num,filename,date,lat,lon,height,Omega,Kappa,Phi1,Phi2\n"
        "1,09_0001.JPG,2023-01-01,32.35,120.05,466,0,0,10,10\n")
    # CSV sınırları bilerek kaydırılmış: georeferans raster'dan okunmalı
    (tmp_path / "satellite_coordinates_range.csv").write_text(
        "mapname,LT_lat_map,LT_lon_map,RB_lat_map,RB_lon_map,region\n"
        f"satellite09.tif,{LT_LAT + 0.01},{LT_LON},{RB_LAT},{RB_LON},Huzhou-3\n")
    reg = load_region(tmp_path, "09")
    assert reg.sat_path.suffix == ".vrt"
    assert (reg.georef.width_px, reg.georef.height_px) == (W, H)
    assert reg.georef.lat_top == pytest.approx(LT_LAT)
    with rasterio.open(reg.sat_path) as ds:
        np.testing.assert_array_equal(ds.read(), full)
