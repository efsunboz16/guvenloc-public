import math

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")

from guvenloc.baselines import (
    features,
    fit_homography,
    match_entropy_bits,
    pose_from_h,
    project,
    select_rotation,
    weighted_dlt,
)
from guvenloc.matcher import MatchSet


def sim_h(bearing_deg, scale, t):
    """Drone → karo: saat yönünde bearing (y aşağı), ölçek, öteleme."""
    a = math.radians(bearing_deg)
    H = np.eye(3)
    H[:2, :2] = scale * np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    H[:2, 2] = t
    return H


def make_ms(H, n=600, out_frac=0.3, noise=0.5, seed=0, wh=(400, 300)):
    rng = np.random.default_rng(seed)
    u = rng.uniform([0, 0], wh, (n, 2))
    v = project(H, u) + rng.normal(0, noise, (n, 2))
    k = int(out_frac * n)
    v[:k] = rng.uniform(0, 1024, (k, 2))
    S = np.tile(np.eye(2), (n, 1, 1)) * noise**2
    return MatchSet(u.astype(np.float32), v.astype(np.float32), np.full(n, 0.9, np.float32),
                    np.zeros_like(S, np.float32), S.astype(np.float32),
                    {"drone_wh": list(wh)})


def test_weighted_dlt_exact():
    H = sim_h(30, 0.5, [300, 200])
    H[2, :2] = [1e-4, -5e-5]
    u = np.random.default_rng(1).uniform(0, 400, (50, 2))
    Hh = weighted_dlt(u, project(H, u), np.random.default_rng(2).uniform(0.1, 1, 50))
    np.testing.assert_allclose(Hh / Hh[2, 2], H / H[2, 2], atol=1e-8)


@pytest.mark.parametrize("bearing", [0, 45, 164, 270])
def test_fit_and_pose(bearing):
    H = sim_h(bearing, 0.4, [512, 512])
    ms = make_ms(H)
    fit = fit_homography(ms, ransac_px=3)
    assert fit["n_inliers"] >= 0.65 * len(ms)
    assert not fit["inliers"][: int(0.3 * len(ms))].mean() > 0.05   # aykırıların çoğu dışarıda
    pose = pose_from_h(fit["H"], (400, 300))
    c_true = project(H, np.array([[199.5, 149.5]]))[0]
    assert math.hypot(pose["cx"] - c_true[0], pose["cy"] - c_true[1]) < 1.0
    d = (pose["bearing"] - bearing + 180) % 360 - 180
    assert abs(d) < 1.0 and pose["scale"] == pytest.approx(0.4, rel=0.02)
    assert not pose["mirrored"]


def test_mirrored_h_is_invalid():
    H = sim_h(0, 0.4, [512, 512]) @ np.diag([-1.0, 1.0, 1.0])
    f = features(make_ms(H, out_frac=0.0), ransac_px=3)
    assert f["n_inliers"] == 0 and not f["valid"]


def test_entropy_and_select():
    assert match_entropy_bits([0.5, 0.5]) == pytest.approx(1.0)
    assert match_entropy_bits([0.999]) < 0.02
    g = pd.DataFrame({"rot": [0, 90, 180, 270], "n_inliers": [10, 50, 20, 5],
                      "entropy": [0.3, 0.5, 0.1, 0.9]})
    assert select_rotation(g, 1).rot == 0
    assert select_rotation(g, 2).rot == 90
    assert select_rotation(g, 3).rot == 180
