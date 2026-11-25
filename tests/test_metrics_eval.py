import math

import numpy as np
import pytest

pytest.importorskip("scipy")

from guvenloc import metrics as M


def test_success_and_median():
    e = np.array([5, 20, 30, np.nan, 8])
    acc = np.array([1, 1, 1, 1, 0], bool)
    assert M.success_at(e, 25) == pytest.approx(3 / 5)          # NaN başarısız
    assert M.success_at(e, 25, acc) == pytest.approx(2 / 5)     # reddedilen başarısız
    assert M.success_among_accepted(e, 25, acc) == pytest.approx(2 / 4)
    assert M.median_error(e) == pytest.approx(14.0)


def test_location_and_heading_error():
    assert M.location_error_m([[3, 4]], [[0, 0]])[0] == pytest.approx(5)
    np.testing.assert_allclose(M.heading_error_deg([359, 10], [1, 350]), [2, 20])


def test_auroc_matches_definition():
    assert M.auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert M.auroc([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0]) == 0.0
    assert M.auroc([0.5, 0.5], [1, 0]) == 0.5
    rng = np.random.default_rng(0)
    s, y = rng.normal(size=300), rng.integers(0, 2, 300)
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in s[y == 1] for b in s[y == 0]])
    assert M.auroc(s, y) == pytest.approx(brute)


def test_far_and_coverage():
    is_pos = np.array([1, 1, 1, 0, 0], bool)
    acc = np.array([1, 1, 0, 1, 0], bool)
    err = np.array([10, 40, 5, np.nan, np.nan])
    assert M.far_neg(acc, is_pos) == 0.5
    assert M.far_loc(acc, is_pos, err) == pytest.approx(1 / 3)
    assert M.coverage([1, 5, 9], [2, 6, 8]) == pytest.approx(2 / 3)
    np.testing.assert_array_equal(M.good_label(is_pos, err), [1, 0, 1, 0, 0])


def test_bins():
    np.testing.assert_array_equal(M.heading_bin([0, 44.9, 45, 359, -1]), [0, 0, 1, 7, 7])
    np.testing.assert_array_equal(M.altitude_bin([300, 400, 900, 1500, 2000, 2500]),
                                  [-1, 0, 1, 2, 2, -1])
    assert math.isnan(M.far_neg([1], [1]))
