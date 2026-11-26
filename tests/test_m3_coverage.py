"""MIMARI 4A – sentetik doğrulama: değiştirilebilir veride FAR-neg ≤ α ve kabul koşullu kapsama ≈ 1 − β.

Heteroskedastik 2B Gauss hata (σ log-normal), σ̂ = σ · gürültü; 200 tekrar × (10.000 kalibrasyon + 10.000 test).
"""
from __future__ import annotations

import numpy as np

from guvenloc import m3_conformal as C

ALPHA, BETA, REPS, N = 0.05, 0.10, 200, 10_000


def draw(rng, n):
    is_pos = rng.random(n) < 0.5
    score = np.where(is_pos, rng.normal(2.0, 1.0, n), rng.normal(0.0, 1.0, n))
    sigma = np.exp(rng.normal(2.5, 0.7, n))                    # ≈ 12 m medyan, geniş yayılım
    sig_hat = sigma * np.exp(rng.normal(0.0, 0.2, n))
    err = sigma * np.hypot(rng.normal(size=n), rng.normal(size=n))
    return score, is_pos, err, sig_hat


def run(mondrian=None, normalize=True, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(REPS):
        s, p, e, sh = draw(rng, N)
        sh_c = sh if normalize else np.ones_like(sh)
        cal = C.fit(s[~p], s[p], e[p], sh_c[p], ALPHA, BETA, mondrian_bins=mondrian)
        s2, p2, e2, sh2 = draw(rng, N)
        r = C.evaluate(cal, s2, p2, e2, sh2 if normalize else np.ones_like(sh2))
        if not normalize:                                     # zor çeyrek σ̂'ye göre tanımlı kalmalı
            acc = cal.accept(s2) & p2
            hard = sh2[acc] >= np.quantile(sh2[acc], 0.75)
            r["coverage_hard"] = float((e2[acc] <= cal.q[0])[hard].mean())
        out.append(r)
    return {k: np.mean([o[k] for o in out]) for k in ("far_neg", "coverage", "coverage_hard")}


def test_order_stat_finite_sample():
    assert C.order_stat(np.arange(1, 10), 0.9) == 9           # ⌈10·0,9⌉ = 9
    assert C.order_stat(np.arange(1, 9), 0.9) == np.inf       # ⌈9·0,9⌉ = 9 > 8


def test_far_and_coverage_marginal():
    m = run()
    assert m["far_neg"] <= ALPHA + 0.001
    assert abs(m["far_neg"] - ALPHA) < 0.002
    assert abs(m["coverage"] - (1 - BETA)) < 0.003


def test_mondrian_hard_quartile():
    m = run(mondrian=4, seed=1)
    assert abs(m["coverage"] - (1 - BETA)) < 0.003
    assert m["coverage_hard"] >= 1 - BETA - 0.01


def test_standard_conformal_undercovers_hard_quartile():
    m = run(normalize=False, seed=2)                            # sabit yarıçap: genelde tutar, zor çeyrekte düşer
    assert abs(m["coverage"] - (1 - BETA)) < 0.003
    assert m["coverage_hard"] < 0.80


def test_sigma_edges_min_group():
    rng = np.random.default_rng(0)
    assert len(C.sigma_edges(rng.random(120), 4, 50)) == 1     # 4 dilim × 30 < 50 → 2 dilim
    assert len(C.sigma_edges(rng.random(40), 4, 50)) == 0
