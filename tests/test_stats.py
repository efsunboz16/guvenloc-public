import numpy as np
import pytest

pytest.importorskip("scipy")

from guvenloc import stats as S


def test_holm_known_values():
    np.testing.assert_allclose(S.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_clopper_pearson_edges():
    lo, hi = S.clopper_pearson(0, 10)
    assert lo == 0 and 0.30 < hi < 0.31          # bilinen değer 0,3085
    lo, hi = S.clopper_pearson(5, 10)
    assert lo == pytest.approx(0.1871, abs=1e-3) and hi == pytest.approx(0.8129, abs=1e-3)


def test_mcnemar():
    a = np.array([1] * 20 + [0] * 2 + [1] * 50, bool)
    b = np.array([0] * 20 + [1] * 2 + [1] * 50, bool)
    r = S.mcnemar(a, b)
    assert r["a_only"] == 20 and r["b_only"] == 2 and r["p"] < 0.001
    assert S.mcnemar(a, a)["p"] == 1.0


def test_wilcoxon_detects_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(10, 1, 100)
    assert S.wilcoxon(x + 1, x + rng.normal(0, 0.1, 100))["p"] < 1e-6
    assert S.wilcoxon(x, x)["p"] == 1.0


def test_cluster_bootstrap_wider_than_iid():
    rng = np.random.default_rng(1)
    clusters = np.repeat(np.arange(10), 50)
    vals = np.repeat(rng.normal(0, 1, 10), 50) + rng.normal(0, 0.1, 500)
    pt, lo, hi = S.cluster_bootstrap_ci(lambda i: vals[i].mean(), clusters, B=500)
    assert lo < pt < hi
    # kümeler içi bağımlılık: GA genişliği iid standart hatanın çok üstünde olmalı
    assert (hi - lo) > 5 * 2 * 1.96 * vals.std() / np.sqrt(500)
