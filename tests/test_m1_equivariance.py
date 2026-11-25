"""Modül 1 (m1_canon) — dönme kuralları ve C8 eşdeğerliği (MIMARI 2B birim testi).

escnn CI'da kurulu değil → ağ testleri atlanır; sunucuda çalışır (CPU yeter).
"""
import numpy as np
import pytest
import torch

from guvenloc import m1_canon as M


def smooth_input(b=2, s=M.IN_SIZE, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, 3, s, s, generator=g)
    x = torch.nn.functional.avg_pool2d(x, 9, 1, 4)
    return x * M.circular_mask(s)


def test_rotate_cw_matches_rot90_clockwise():
    x = smooth_input()
    m = M.circular_mask(x.shape[-1])
    for k in (1, 2, 3):
        a = M.rotate_cw(x, torch.full((len(x),), 90.0 * k))
        assert torch.allclose(a, M.rotate_cw_90k(x, k) * m, atol=1e-4)


def test_heading_rule_and_bins():
    assert M.heading_after_cw(30.0, 90.0) == pytest.approx(300.0)
    assert M.heading_after_cw(350.0, -20.0) == pytest.approx(10.0)
    assert list(M.to_bins([0.0, 9.99, 10.0, 359.9])) == [0, 0, 1, 35]


def test_soft_targets_peak_and_norm():
    t = M.soft_targets(torch.tensor([123.0, 2.0]))
    assert torch.allclose(t.sum(1), torch.ones(2))
    assert t.argmax(1).tolist() == [12, 0]
    assert t[1, 35] > t[1, 2]  # dairesel: 2° → 355° bölmesi 25° bölmesinden yakın


def test_predict_deg_subbin():
    th = np.deg2rad(M.bin_centers_deg())
    for true in (0.3, 123.4, 359.2, 187.0):
        logits = torch.tensor(8 * np.cos(th - np.deg2rad(true)), dtype=torch.float32)[None]
        assert float(M.angular_error_deg(M.predict_deg(logits), torch.tensor([true]))) < 1.0


def test_fourier_interp_shift_is_exact():
    """8 yön logitinde 2 adım kayma (90°) → 36 bölmede tam 9 bölme kayma.

    45° = 4,5 bölme: 36 bölmeli ızgarada tam sayı değil; 45°'lik kayma sürekli açıda (alt-bölme tahmini) tam.
    """
    P = M.fourier_interp_matrix()
    v = torch.randn(3, M.N_GROUP)
    a = v @ P.T
    b = v.roll(2, dims=1) @ P.T
    assert torch.allclose(b, a.roll(9, dims=1), atol=1e-5)
    assert torch.allclose(P.sum(1), torch.ones(M.N_BINS), atol=1e-6)
    # 45°: trigonometrik polinomun kendisi tam 45° kayar (bölme merkezlerinde değil, açıda)
    th = np.deg2rad(np.arange(0, 360, 1.0))
    def f(vv, t):
        k = np.arange(M.N_GROUP)
        d = t[:, None] - np.deg2rad(k * 45.0)[None]
        m = 1 + sum(2 * np.cos(j * d) for j in range(1, 4)) + np.cos(4 * d)
        return (m / M.N_GROUP) @ vv
    vv = v[0].numpy()
    assert np.allclose(f(np.roll(vv, 1), th), f(vv, th - np.deg2rad(45.0)), atol=1e-5)


def test_c8_equivariance_90():
    pytest.importorskip("escnn")
    torch.manual_seed(0)
    net = M.C8Canon(widths=(4, 8, 8, 8, 8)).eval()
    x = smooth_input()
    with torch.no_grad():
        l0 = net(x)
        for k in (1, 2, 3):
            lk = net(M.rotate_cw_90k(x, k))
            # içerik saat yönünde 90k° → φ − 90k° → 36 bölmede −9k kayma
            exp = l0.roll(-9 * k, dims=1)
            assert torch.allclose(lk, exp, atol=1e-4 * float(l0.abs().max()) + 1e-6), k
