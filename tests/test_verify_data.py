"""scripts/11_verify_data.py için sahte UAV-VisLoc düzeniyle duman testi."""

import importlib.util
import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_bounds

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "11_verify_data.py"


def load_script():
    spec = importlib.util.spec_from_file_location("verify_data", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_fake(root: Path, regions=("01", "02"), n=3, drop_phi_in="02"):
    root.mkdir(parents=True)
    rows = []
    for rid in regions:
        r = root / rid
        (r / "drone").mkdir(parents=True)
        recs = []
        for i in range(1, n + 1):
            name = f"{rid}_{i:04d}.JPG"
            (r / "drone" / name).write_bytes(b"\xff\xd8fake")
            recs.append({"filename": name, "lat": 30.0 + i * 1e-4, "lon": 100.0, "height": 500.0,
                         "Phi1": np.nan if rid == drop_phi_in else 10.0 * i})
        pd.DataFrame(recs).to_csv(r / f"{rid}.csv", index=False)
        with rasterio.open(r / f"satellite{rid}.tif", "w", driver="GTiff", width=8, height=6, count=3,
                           dtype="uint8", crs="EPSG:4326",
                           transform=from_bounds(100, 30, 100.01, 30.01, 8, 6)) as ds:
            ds.write(np.zeros((3, 6, 8), np.uint8))
        rows.append({"mapname": f"satellite{rid}.tif", "LT_lat": 30.01, "LT_lon": 100, "RB_lat": 30, "RB_lon": 100.01})
    pd.DataFrame(rows).to_csv(root / "satellite_coordinates_range.csv", index=False)


def test_verify_ok_and_detects_missing_heading(tmp_path, monkeypatch):
    root = tmp_path / "UAV-VisLoc"
    make_fake(root)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["x", str(tmp_path), "--expect-frames", "6", "--expect-maps", "2", "--checksums"])
    assert load_script().main() == 0  # tek üst klasöre iniyor mu?

    rep = json.loads((tmp_path / "results/faz1" / f"verify_{tmp_path.name}.json").read_text(encoding="utf-8"))
    assert rep["totals"] == {"regions": 2, "frames": 6, "maps": 2}
    r02 = next(r for r in rep["regions"] if r["region"] == "02")
    assert r02["missing_ratio"]["Phi1"] == 1.0
    assert r02["raster"]["width"] == 8
    ck = (tmp_path / "docs/checksums" / f"{tmp_path.name}.sha256").read_text().splitlines()
    assert len(ck) == 2 * (3 + 2) + 1


def test_verify_flags_csv_disk_mismatch(tmp_path, monkeypatch):
    root = tmp_path / "ds"
    make_fake(root, regions=("01",))
    (root / "01" / "drone" / "01_0002.JPG").unlink()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["x", str(root)])
    assert load_script().main() == 1
