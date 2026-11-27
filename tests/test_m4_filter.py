"""MIMARI 5A – sentetik ölçümlerle filtre iskeleti: Gauss tepe + %10–30 yanlış tepe + rastgele ret.

Yörünge: 300 adım × ≈ 80 m, yavaş dönen yön. Her karede 1 km'lik karo (4 m hücre), doğru tepe σ = 10 m ya da
(oran p_wrong) karoda rastgele bir yerde yanlış tepe. M3 kabulü: doğru tepede 0,6, yanlışta 0,1 olasılık.
Hareket ölçümü: gerçek (ileri, sağ) + %10 gürültü, Δθ + 0,5°.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from guvenloc import m4_filter as F

TILE_M, CELL_M, MPP = 1000.0, 4.0, 1.0


def grid_at(center, peak, sigma=10.0):
    n = int(TILE_M / CELL_M)
    meas = F.GridMeas(np.zeros((n, n)), center, int(TILE_M / MPP), MPP, CELL_M)
    cx, cy = meas.to_cell(*peak)
    yy, xx = np.mgrid[0:n, 0:n]
    P = np.exp(-0.5 * ((xx - cx) ** 2 + (yy - cy) ** 2) * CELL_M ** 2 / sigma ** 2)
    meas.P = P / P.sum()
    return meas


def simulate(p_wrong: float, seed: int = 0, steps: int = 300, gate: bool = True):
    rng = np.random.default_rng(seed)
    pf = F.ParticleFilter(F.FilterCfg(n_particles=1500, seed=seed))
    pos, th = np.zeros(2), 30.0
    e_single, e_filt, wrongs = [], [], []
    for k in range(steps):
        if k:
            dth = 2.0 * math.sin(k / 40)
            fwd, right = 80.0, rng.normal(0, 5)
            t = math.radians(th)
            pos = pos + [fwd * math.sin(t) + right * math.cos(t), fwd * math.cos(t) - right * math.sin(t)]
            th = (th + dth) % 360
            if pf.started:
                pf.predict(fwd * (1 + rng.normal(0, 0.1)), right * (1 + rng.normal(0, 0.1)), dth + rng.normal(0, 0.5))
        center = pos + rng.uniform(-150, 150, 2)
        wrong = rng.random() < p_wrong
        peak = center + rng.uniform(-450, 450, 2) if wrong else pos + rng.normal(0, 10, 2)
        acc = rng.random() < (0.1 if wrong else 0.6)
        meas = grid_at(tuple(center), tuple(peak))
        e_single.append(float(np.hypot(*(peak - pos))))
        wrongs.append(wrong)
        if not pf.started:
            if acc:
                pf.init(peak[0], peak[1], 40.0, th + rng.normal(0, 2))
            e_filt.append(e_single[-1])
            continue
        if acc or not gate:
            pf.update(meas, th + rng.normal(0, 2))
        est = pf.estimate()
        e_filt.append(float(np.hypot(est["e"] - pos[0], est["n"] - pos[1])))
    return np.array(e_single), np.array(e_filt), np.array(wrongs)


@pytest.mark.parametrize("p_wrong", [0.1, 0.3])
def test_filter_recovers_wrong_peak_frames(p_wrong):
    es, ef, wr = simulate(p_wrong)
    assert np.median(ef[wr]) < 30.0                       # tek kare bu karelerde yüzlerce metre
    assert np.median(es[wr]) > 200.0
    assert np.median(ef) <= np.median(es) * 1.1           # iyi karelerde kötüleştirmez (tek kare σ = 10 m)


def test_median_reduction_when_wrong_peaks_frequent():
    es, ef, _ = simulate(0.5, seed=2)                    # tek kare medyanı yanlış tepelerce belirlenir
    assert np.median(ef) <= 0.8 * np.median(es)


def test_gate_off_still_robust():
    _, ef_gate, _ = simulate(0.3, seed=1)
    _, ef_nogate, _ = simulate(0.3, seed=1, gate=False)  # ε tabanı tek yanlış tepeyi tolere eder
    assert np.median(ef_nogate) < 30.0 and np.median(ef_gate) < 30.0


def test_grid_roundtrip_and_lookup():
    m = grid_at((100.0, -50.0), (130.0, -20.0))
    x_px, y_px = 530.0 - 0.5 + 0.0, 500 - 30.0 - 0.5
    assert np.allclose(m.to_en(x_px, y_px), (130.0, -20.0))
    v = m.lookup(np.array([130.0, 130.0 + 600]), np.array([-20.0, -20.0]))
    assert v[0] > 0.95 and v[1] == 0.0


def test_weighted_quantile_and_resample():
    pf = F.ParticleFilter(F.FilterCfg(n_particles=1000))
    pf.init(0.0, 0.0, 100.0, 0.0)
    est = pf.estimate()
    assert 85 < est["spread_q90_m"] < 100                    # diskte düzgün: r_90 = 100·√0,9 ≈ 94,9
    pf.w = np.zeros(1000)
    pf.w[0] = 1.0
    pf.resample()
    assert np.allclose(pf.x[:, :2], pf.x[0, :2])
