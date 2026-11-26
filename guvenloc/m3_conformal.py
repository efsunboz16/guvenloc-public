"""Modül 3 – split conformal kalibrasyon (MIMARI.md Faz 4).

Ret eşiği τ (4B): kalibrasyon negatiflerinin skorlarından
    τ = S⁻_(k),  k = ⌈(n⁻ + 1)(1 − α)⌉   (k-ıncı en küçük),   kabul ⇔ S > τ
Değiştirilebilirlik altında yeni bir negatifin kabul olasılığı ≤ α. Eşitlik bozmak için kabul kesin büyüklükle
(S > τ); sürekli skorlarda plandaki S ≥ τ ile aynıdır.

Güven bölgesi q̂ (4C): kalibrasyonda pozitif ∧ kabul edilmiş çiftlerde r = ‖gt − x̂‖ / σ̂,
    q̂ = r_(k),  k = ⌈(n + 1)(1 − β)⌉,   yarıçap = q̂ · σ̂
Kapsama garantisi kabul koşulludur: P(‖gt − x̂‖ ≤ q̂σ̂ | pozitif, kabul) ≥ 1 − β.

Mondrian (4F): q̂ her σ̂ diliminde ayrı; dilim sınırları kalibrasyonun kabul edilmiş pozitiflerinin σ̂
yüzdeliklerinden, grup başına n < min_group ise dilim sayısı azaltılır. `sigma` yerine sabit 1 verilirse
standart (σ̂ ile ölçeklenmemiş, sabit yarıçaplı) split conformal elde edilir.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def order_stat(x: np.ndarray, level: float) -> float:
    """⌈(n + 1)·level⌉-inci en küçük değer; n yetmezse +∞ (conformal sonlu örnek düzeltmesi)."""
    x = np.sort(np.asarray(x, float))
    k = math.ceil((len(x) + 1) * level)
    return float(x[k - 1]) if 1 <= k <= len(x) else math.inf


def reject_threshold(neg_scores: np.ndarray, alpha: float) -> float:
    return order_stat(neg_scores, 1.0 - alpha)


def radius_quantile(r: np.ndarray, beta: float) -> float:
    return order_stat(r, 1.0 - beta)


def sigma_edges(sigma: np.ndarray, bins: int, min_group: int) -> np.ndarray:
    """σ̂ yüzdelik sınırları (iç sınırlar); her grupta ≥ min_group örnek kalana dek dilim sayısı azaltılır."""
    sigma = np.asarray(sigma, float)
    for b in range(bins, 0, -1):
        edges = np.quantile(sigma, np.linspace(0, 1, b + 1)[1:-1]) if b > 1 else np.array([])
        if np.bincount(np.searchsorted(edges, sigma, side="right"), minlength=b).min() >= min_group:
            return edges
    return np.array([])


@dataclass
class Calibration:
    tau: float
    q: np.ndarray                      # dilim başına q̂ (Mondrian yoksa tek eleman)
    edges: np.ndarray = field(default_factory=lambda: np.array([]))
    n_neg: int = 0
    n_pos_acc: np.ndarray = field(default_factory=lambda: np.array([]))

    def group(self, sigma: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.edges, np.asarray(sigma, float), side="right")

    def accept(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, float) > self.tau

    def radius(self, sigma: np.ndarray) -> np.ndarray:
        sigma = np.asarray(sigma, float)
        return self.q[self.group(sigma)] * sigma


def fit(neg_scores, pos_scores, pos_err, pos_sigma, alpha: float, beta: float,
        mondrian_bins: int | None = None, min_group: int = 50) -> Calibration:
    """Kalibrasyon: negatiflerden τ, pozitif ∧ kabul edilmişlerden (dilim başına) q̂."""
    tau = reject_threshold(neg_scores, alpha)
    acc = np.asarray(pos_scores, float) > tau
    err, sig = np.asarray(pos_err, float)[acc], np.asarray(pos_sigma, float)[acc]
    edges = sigma_edges(sig, mondrian_bins, min_group) if mondrian_bins else np.array([])
    g = np.searchsorted(edges, sig, side="right")
    q = np.array([radius_quantile(err[g == k] / sig[g == k], beta) for k in range(len(edges) + 1)])
    return Calibration(tau=tau, q=q, edges=edges, n_neg=len(np.asarray(neg_scores)),
                       n_pos_acc=np.bincount(g, minlength=len(edges) + 1))


def evaluate(cal: Calibration, scores, is_pos, err, sigma, hard_frac: float = 0.25) -> dict:
    """FAR-neg, kabul koşullu kapsama (genel ve σ̂'ye göre en belirsiz hard_frac dilimi), ortalama yarıçap."""
    scores, is_pos = np.asarray(scores, float), np.asarray(is_pos, bool)
    err, sigma = np.asarray(err, float), np.asarray(sigma, float)
    acc = cal.accept(scores)
    ap = acc & is_pos
    rad = cal.radius(sigma)
    cov = err[ap] <= rad[ap]
    hard = sigma[ap] >= np.quantile(sigma[ap], 1 - hard_frac) if ap.sum() else np.zeros(0, bool)
    return {"far_neg": float(acc[~is_pos].mean()), "n_neg": int((~is_pos).sum()),
            "k_neg_acc": int(acc[~is_pos].sum()), "accept_pos": float(acc[is_pos].mean()),
            "n_pos_acc": int(ap.sum()), "coverage": float(cov.mean()) if ap.sum() else math.nan,
            "k_cov": int(cov.sum()), "coverage_hard": float(cov[hard].mean()) if hard.sum() else math.nan,
            "n_hard": int(hard.sum()), "k_cov_hard": int(cov[hard].sum()),
            "radius_mean_m": float(rad[ap].mean()) if ap.sum() else math.nan,
            "radius_median_m": float(np.median(rad[ap])) if ap.sum() else math.nan}
