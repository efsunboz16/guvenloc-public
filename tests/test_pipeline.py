"""4G – pipeline.py: tek satır öznitelik vektörü 61_m2_score.feat_matrix ile aynı; karar ve enlem/boylam dönüşümü."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from guvenloc import pipeline as P
from guvenloc.geo import EARTH_R

ROOT = Path(__file__).resolve().parents[1]
FEATS = ["log1p_support", "support_frac", "rho", "spread_m", "entropy_bits", "log1p_sum_c", "frac_dropped",
         "log1p_inliers"]


def _s61():
    spec = importlib.util.spec_from_file_location("m2s61_t", ROOT / "scripts/61_m2_score.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["m2s61_t"] = mod
    spec.loader.exec_module(mod)
    return mod


def _rows(n=50, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"support": rng.gamma(2, 50, n), "sum_c": rng.gamma(3, 500, n), "support_frac": rng.random(n),
                       "rho": rng.random(n) * 3, "spread_m": rng.gamma(2, 5, n), "entropy_bits": rng.random(n) * 8,
                       "frac_dropped": rng.random(n) * 0.2, "n_inliers": rng.integers(0, 2000, n),
                       "x": rng.random(n) * 1000, "y": rng.random(n) * 1000, "theta": rng.random(n) * 360})
    df.loc[3, "spread_m"] = np.nan
    return df


def test_feat_vector_matches_feat_matrix():
    df = _rows()
    X = _s61().feat_matrix(df, FEATS)
    fill = float(df.entropy_bits.max())
    for i, f in enumerate(df.to_dict("records")):
        assert np.allclose(P.feat_vector(f, FEATS, fill), X[i])


def test_decide_threshold_and_radius():
    df = _rows()
    params = {"tau": 0.0, "q": [3.0], "edges": [], "sigma_floor_m": 2.0, "entropy_fill": 8.0, "volume_cfg": {},
              "score_model": {"features": FEATS, "w": [0.5] + [0.1] * len(FEATS), "mu": [0.0] * len(FEATS),
                              "sd": [1.0] * len(FEATS)}}
    loc = P.Localizer(params)
    for f in df.to_dict("records"):
        d = loc.decide(f, 90, 1.0)
        assert d.accept == (d.score > 0.0)
        sig = 50.0 if not np.isfinite(f["spread_m"]) else max(f["spread_m"], 2.0)
        assert math.isclose(d.sigma_hat, sig) and math.isclose(d.radius_m, 3.0 * sig)


def test_tile_px_to_latlon():
    lat0, lon0, size, px = 30.0, 116.0, 1000.0, 1000
    assert np.allclose(P.tile_px_to_latlon(px / 2 - 0.5, px / 2 - 0.5, lat0, lon0, size, px), (lat0, lon0))
    lat, lon = P.tile_px_to_latlon(px - 0.5, px / 2 - 0.5, lat0, lon0, size, px)   # sağ kenar: 500 m doğu
    assert math.isclose(math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0)), 500.0, rel_tol=1e-9)
    lat, _ = P.tile_px_to_latlon(px / 2 - 0.5, -0.5, lat0, lon0, size, px)           # üst kenar: 500 m kuzey
    assert math.isclose(math.radians(lat - lat0) * EARTH_R, 500.0, rel_tol=1e-9)
