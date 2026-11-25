"""İstatistiksel testler (MIMARI.md §1F, §6A).

- Uçuş (küme) bazlı bootstrap güven aralığı
- McNemar (eşleştirilmiş oranlar), Wilcoxon işaretli sıra (eşleştirilmiş hatalar)
- Holm çoklu karşılaştırma düzeltmesi
- Clopper–Pearson binom güven aralığı
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats as st


def cluster_bootstrap_ci(
    stat_fn: Callable[[np.ndarray], float],
    clusters: Sequence,
    n_items: int | None = None,
    B: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Küme bootstrap yüzdelik GA.

    stat_fn: örnek indisleri dizisi alır, istatistiği döndürür (ör. lambda i: success[i].mean()).
    clusters: her örneğin küme (uçuş) kimliği. Kümeler yerine konarak örneklenir.
    Döndürür: (nokta tahmini, alt, üst).
    """
    clusters = np.asarray(clusters)
    n = len(clusters) if n_items is None else n_items
    ids, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(ids))]
    rng = np.random.default_rng(seed)
    point = float(stat_fn(np.arange(n)))
    vals = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(ids), len(ids))
        vals[b] = stat_fn(np.concatenate([members[k] for k in pick]))
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def mcnemar(a_correct, b_correct) -> dict:
    """Kesin (binom) McNemar testi. a/b: aynı örneklerde iki yöntemin doğru/yanlış dizileri."""
    a = np.asarray(a_correct, bool)
    b = np.asarray(b_correct, bool)
    n01 = int((a & ~b).sum())  # A doğru, B yanlış
    n10 = int((~a & b).sum())  # A yanlış, B doğru
    n = n01 + n10
    p = 1.0 if n == 0 else float(st.binomtest(n01, n, 0.5).pvalue)
    return {"a_only": n01, "b_only": n10, "p": p}


def wilcoxon(a, b) -> dict:
    """Eşleştirilmiş Wilcoxon işaretli sıra testi (a − b). NaN çiftleri atılır."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if not np.any(d != 0):
        return {"n": int(m.sum()), "median_diff": 0.0, "p": 1.0}
    r = st.wilcoxon(a[m], b[m])
    return {"n": int(m.sum()), "median_diff": float(np.median(d)), "p": float(r.pvalue)}


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm–Bonferroni düzeltilmiş p değerleri (girdi sırasıyla)."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Kesin binom GA."""
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(st.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(st.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi
