import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("yaml")

from guvenloc.geo import GeoRef
from guvenloc.pairs import (
    PAIR_COLUMNS,
    RegionInfo,
    Split,
    enu_offset_m,
    load_split,
    make_pairs,
    offset_latlon,
    tile_in_map_fraction,
)

ROOT = Path(__file__).resolve().parents[1]
CFG = {
    "seed": 0, "version": "t", "tile_size_m": 400, "tile_px": 256, "tile_offset_m": 100,
    "hard_neg_per_pos": 1, "easy_neg_per_pos": 1, "hard_neg_min_dist_m": 800,
    "hard_neg_fallback_dist_m": 400, "neg_min_tile_in_map": 0.9, "max_tries": 500,
    "camera": {"image_width_px": 4000, "footprint_over_agl": 1.0},
    "ground_elevation_m": {"02": 100},
}


def region(rid, lat0, n=5, size_deg=0.03):
    ref = GeoRef(lat0 + size_deg, 120.0, lat0, 120.0 + size_deg, 3000, 3000)
    frames = pd.DataFrame({
        "num": range(1, n + 1), "filename": [f"{rid}_{i:04d}.JPG" for i in range(1, n + 1)],
        "lat": lat0 + size_deg / 2, "lon": 120.0 + size_deg / 2, "height": 500.0,
        "Phi1": 30.0,
    })
    return RegionInfo(rid, frames, ref)


REGIONS = {"01": region("01", 30.0), "02": region("02", 31.0), "03": region("03", 32.0),
           "04": region("04", 33.0)}
SPLIT = Split({"01": "train", "02": "train", "03": "train", "04": "test"},
              frozenset(), frozenset({"01_0003.JPG"}),
              {"01": "g", "02": "g", "03": "03", "04": "04"}, {})


def test_real_split_config_is_valid():
    sp = load_split(ROOT / "configs" / "split.yaml")
    assert sp.group_of["08"] == sp.group_of["09"]
    assert sp.region_split["08"] == sp.region_split["09"]
    assert "07" in sp.excluded_regions and "07" not in sp.region_split
    assert set(sp.terrain) == set(sp.region_split)


def test_offset_roundtrip():
    lat, lon = offset_latlon(30.0, 120.0, 123.0, -45.0)
    de, dn = enu_offset_m(30.0, 120.0, lat, lon)
    assert de == pytest.approx(123.0, abs=1e-3) and dn == pytest.approx(-45.0, abs=1e-3)


def test_tile_in_map_fraction():
    ref = REGIONS["01"].georef
    assert tile_in_map_fraction(ref, 30.015, 120.015, 400) == pytest.approx(1.0)
    assert tile_in_map_fraction(ref, 30.0, 120.015, 400) == pytest.approx(0.5, abs=0.01)


def test_pairs_deterministic_and_valid():
    a = make_pairs(REGIONS, SPLIT, CFG)
    b = make_pairs(REGIONS, SPLIT, CFG)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == PAIR_COLUMNS
    assert a.pair_id.is_unique
    assert "01_0003_pos" not in set(a.pair_id)          # dışlanan kare

    pos = a[a.neg_type == "pos"]
    assert (pos.tile_offset_m <= CFG["tile_offset_m"]).all()
    # GT pikseli: karo merkezinden GT'ye öteleme, tile m/px ile
    mpp = CFG["tile_size_m"] / CFG["tile_px"]
    for r in pos.itertuples():
        de, dn = enu_offset_m(r.tile_lat, r.tile_lon, r.drone_lat, r.drone_lon)
        assert r.gt_x == pytest.approx(CFG["tile_px"] / 2 + de / mpp, abs=0.05)
        assert r.gt_y == pytest.approx(CFG["tile_px"] / 2 - dn / mpp, abs=0.05)
    # ölçek: AGL = height - zemin; drone m/px = AGL / 4000
    r02 = pos[pos.region == "02"].iloc[0]
    assert r02.agl_m == 400 and r02.scale_s == pytest.approx((400 / 4000) / mpp)

    hard = a[a.neg_type == "hard"]
    assert (hard.neg_dist_m >= CFG["hard_neg_fallback_dist_m"]).all()
    assert (hard.tile_region == hard.region).all()

    easy = a[a.neg_type == "easy"]
    # kolay negatif: aynı küme, başka grup → 01/02 yalnızca 03'ten; 03 yalnızca 01/02'den
    assert set(easy[easy.region.isin(["01", "02"])].tile_region) == {"03"}
    assert set(easy[easy.region == "03"].tile_region) <= {"01", "02"}
    assert easy[easy.region == "04"].empty                # test kümesinde başka bölge yok
    assert (a[a.label == "neg"].tile_in_map >= 0.9).all()


def test_seed_changes_pairs():
    a = make_pairs(REGIONS, SPLIT, CFG)
    b = make_pairs(REGIONS, SPLIT, {**CFG, "seed": 1})
    assert not np.allclose(a.tile_lat.to_numpy(), b.tile_lat.to_numpy())
    assert math.isclose(len(a), len(b))
