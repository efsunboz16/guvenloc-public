"""Taban çizgileri — ablasyon satırları 1–3 (MIMARI.md 1F).

| Satır | Yöntem | Konum kararı | Ret kararı |
| 1 | RoMa v2 (0°) + kovaryans ağırlıklı RANSAC homografi | drone merkezi H ile karoya | inlier ≥ τ |
| 2 | 4 yön + en çok inlier | en çok inlier'lı yönün H'si | aynı |
| 3 | 4 yön + en düşük entropi [14] | en düşük eşleşme entropili yön | aynı |

Homografi: RANSAC (USAC_MAGSAC) → inlier'larda IRLS, ağırlık w_i = c_i / tr(Σ_i) × Huber(r_i).
Σ_i = Σ_B + |det A| Σ_A (A: H'nin u_i'deki yerel doğrusal kısmı; tek yönlü ayarlarda Σ_A = 0).
Eşleşme entropisi: örtüşme güveni c'nin ortalama ikili entropisi (bit) — [14]'ün bizim
işlemselleştirmemiz: düşük entropi = RoMa'nın eşleşmeden emin olduğu yön.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from guvenloc.matcher import MatchSet

ROWS = {1: "RoMa v2 (0°) + RANSAC", 2: "4 yön + en çok inlier", 3: "4 yön + en düşük entropi"}


def _normalize(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    ws = w / w.sum()
    c = (p * ws[:, None]).sum(0)
    d = np.sqrt((ws * ((p - c) ** 2).sum(1)).sum())
    s = math.sqrt(2) / max(d, 1e-12)
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])


def weighted_dlt(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> np.ndarray | None:
    """Ağırlıklı, Hartley normalizasyonlu DLT homografi (u → v)."""
    if len(u) < 4 or w.sum() <= 0:
        return None
    Tu, Tv = _normalize(u, w), _normalize(v, w)
    un = (np.c_[u, np.ones(len(u))] @ Tu.T)[:, :2]
    vn = (np.c_[v, np.ones(len(v))] @ Tv.T)[:, :2]
    x, y, X, Y = un[:, 0], un[:, 1], vn[:, 0], vn[:, 1]
    z, o = np.zeros_like(x), np.ones_like(x)
    r1 = np.stack([-x, -y, -o, z, z, z, X * x, X * y, X], 1)
    r2 = np.stack([z, z, z, -x, -y, -o, Y * x, Y * y, Y], 1)
    sw = np.sqrt(w)[:, None]
    A = np.vstack([r1 * sw, r2 * sw])
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Tv) @ Hn @ Tu
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else None


def project(H: np.ndarray, p: np.ndarray) -> np.ndarray:
    q = np.c_[p, np.ones(len(p))] @ H.T
    return q[:, :2] / q[:, 2:3]


def local_linear(H: np.ndarray, p: np.ndarray) -> np.ndarray:
    """H'nin p noktalarındaki Jacobian'ı (N,2,2)."""
    x, y = p[:, 0], p[:, 1]
    den = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    nx = H[0, 0] * x + H[0, 1] * y + H[0, 2]
    ny = H[1, 0] * x + H[1, 1] * y + H[1, 2]
    J = np.empty((len(p), 2, 2))
    J[:, 0, 0] = (H[0, 0] * den - nx * H[2, 0]) / den**2
    J[:, 0, 1] = (H[0, 1] * den - nx * H[2, 1]) / den**2
    J[:, 1, 0] = (H[1, 0] * den - ny * H[2, 0]) / den**2
    J[:, 1, 1] = (H[1, 1] * den - ny * H[2, 1]) / den**2
    return J


def match_entropy_bits(c: np.ndarray) -> float:
    c = np.clip(np.asarray(c, np.float64), 1e-6, 1 - 1e-6)
    return float(np.mean(-(c * np.log2(c) + (1 - c) * np.log2(1 - c))))


def fit_homography(ms: MatchSet, ransac_px: float = 5.0, irls_iters: int = 3,
                   max_iters: int = 5000, seed: int = 0) -> dict:
    """Kovaryans ağırlıklı sağlam homografi. Dönüş: H, inlier maskesi ve özellikler."""
    u = ms.u.astype(np.float64)
    v = ms.v.astype(np.float64)
    out = {"H": None, "n_matches": len(u), "n_inliers": 0, "inliers": np.zeros(len(u), bool)}
    if len(u) < 8:
        return out
    cv2.setRNGSeed(seed)
    H, mask = cv2.findHomography(u, v, cv2.USAC_MAGSAC, ransacReprojThreshold=ransac_px,
                                 maxIters=max_iters, confidence=0.999)
    if H is None or mask is None or mask.sum() < 8:
        return out
    inl = mask.ravel().astype(bool)
    trB = np.trace(ms.Sigma_B, axis1=1, axis2=2).astype(np.float64)
    trA = np.trace(ms.Sigma_A, axis1=1, axis2=2).astype(np.float64)
    c = ms.c.astype(np.float64)
    for _ in range(irls_iters):
        detA = np.abs(np.linalg.det(local_linear(H, u[inl])))
        w_cov = c[inl] / np.maximum(trB[inl] + detA * trA[inl], 1e-6)
        r = np.linalg.norm(project(H, u[inl]) - v[inl], axis=1)
        huber = np.where(r <= ransac_px, 1.0, ransac_px / np.maximum(r, 1e-12))
        Hn = weighted_dlt(u[inl], v[inl], w_cov * huber)
        if Hn is None:
            break
        H = Hn
        r_all = np.linalg.norm(project(H, u) - v, axis=1)
        inl_new = r_all < ransac_px
        if inl_new.sum() < 8:
            break
        inl = inl_new
    out.update(H=H, n_inliers=int(inl.sum()), inliers=inl)
    return out


def pose_from_h(H: np.ndarray, drone_wh) -> dict:
    """Drone merkezinin karo pikseli, görüntü üst kenarının pusula yönü ve ölçek."""
    w, h = drone_wh
    c0 = np.array([[(w - 1) / 2, (h - 1) / 2]])
    d = 0.05 * h
    p = project(H, np.vstack([c0, c0 + [0, -d], c0 + [d, 0]]))
    up, right = p[1] - p[0], p[2] - p[0]
    bearing = math.degrees(math.atan2(up[0], -up[1])) % 360.0
    det = up[0] * right[1] - up[1] * right[0]  # birim H'de d² > 0; yansımada < 0
    scale = math.sqrt(abs(det)) / d
    return {"cx": float(p[0, 0]), "cy": float(p[0, 1]), "bearing": bearing, "scale": scale,
            "mirrored": bool(det < 0)}


def features(ms: MatchSet, ransac_px: float = 5.0, seed: int = 0,
             scale_range: tuple[float, float] = (0.01, 100.0)) -> dict:
    """Tek (çift, yön) için taban çizgisi özellikleri. Geçersiz H → n_inliers = 0."""
    fit = fit_homography(ms, ransac_px=ransac_px, seed=seed)
    f = {"n_matches": fit["n_matches"], "n_inliers": fit["n_inliers"],
         "entropy": match_entropy_bits(ms.c) if len(ms) else 1.0,
         "mean_c": float(np.mean(ms.c)) if len(ms) else 0.0,
         "cx": np.nan, "cy": np.nan, "bearing": np.nan, "scale": np.nan, "valid": False}
    if fit["H"] is not None:
        pose = pose_from_h(fit["H"], ms.meta["drone_wh"])
        ok = (not pose["mirrored"]) and scale_range[0] < pose["scale"] < scale_range[1] \
            and np.isfinite([pose["cx"], pose["cy"]]).all()
        f.update({k: pose[k] for k in ("cx", "cy", "bearing", "scale")}, valid=bool(ok))
        if not ok:
            f["n_inliers"] = 0
    f["inlier_ratio"] = f["n_inliers"] / max(f["n_matches"], 1)
    return f


def select_rotation(group, row: int):
    """Bir çiftin 4 yön satırından taban çizgisi satırına göre seçilen satır (pandas)."""
    if row == 1:
        g = group[group.rot == 0]
        return g.iloc[0] if len(g) else None
    if row == 2:
        return group.loc[group.n_inliers.idxmax()]
    if row == 3:
        return group.loc[group.entropy.idxmin()]
    raise ValueError(row)
