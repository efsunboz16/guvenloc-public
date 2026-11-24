"""RoMa v2 tekrarlama yardımcıları (MIMARI.md 1D): MegaDepth-1500, C4 türetimi.

C4 (MegaDepth-C4): i. çiftin ikinci görüntüsü (i mod 4)·90° döndürülür; iç parametre K ve
poz (R, t; dünya → kamera) AYNI dönmeyle güncellenir.

np.rot90(img, k) saat yönünün TERSİNE k×90° döndürür. k = 1 için (W×H görüntü, piksel
kenarı kuralı): x' = y, y' = W − x. Kamera koordinatında X' = Y, Y' = −X, Z' = Z:
  K' = [[fy, 0, cy], [0, fx, W − cx], [0, 0, 1]],  R' = Rz R,  t' = Rz t,
  Rz = [[0, 1, 0], [−1, 0, 0], [0, 0, 1]].
k kez uygulanır (her adımda W, H yer değiştirir).
"""
from __future__ import annotations

import numpy as np

# romav2 2.0.1 geometry.pose_auc, NumPy 2.5'te kaldırılan np.trapz'ı çağırıyor (requirements: numpy 2.5.4).
# np.trapz = np.trapezoid'in eski adı (aynı fonksiyon) → romav2'nin hem kendi Mega1500.benchmark'ı hem
# bizim döngümüz için geri eklenir. Sayısal sonuç değişmez.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

RZ90 = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def rot90_camera(K: np.ndarray, R: np.ndarray, t: np.ndarray, W: float, H: float, k: int):
    """np.rot90(img, k) sonrası (K', R', t', W', H')."""
    K, R, t = K.astype(np.float64).copy(), R.astype(np.float64).copy(), t.astype(np.float64).copy()
    for _ in range(k % 4):
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        K = np.array([[fy, 0.0, cy], [0.0, fx, W - cx], [0.0, 0.0, 1.0]])
        R, t = RZ90 @ R, RZ90 @ t
        W, H = H, W
    return K, R, t, W, H


def rot90_points(pts: np.ndarray, W: float, H: float, k: int) -> np.ndarray:
    """Piksel kenarı kuralında noktaların np.rot90(img, k) sonrası konumu."""
    p = np.asarray(pts, np.float64).copy()
    for _ in range(k % 4):
        p = np.stack([p[:, 1], W - p[:, 0]], 1)
        W, H = H, W
    return p


MEGA1500_SCENES = ("0015_0.1_0.3.npz", "0015_0.3_0.5.npz", "0022_0.1_0.3.npz",
                   "0022_0.3_0.5.npz", "0022_0.5_0.7.npz")


def load_mega1500(data_root: str, limit: int | None = None) -> list[dict]:
    """romav2.benchmarks.mega1500 sahneleri; limit → sahne başına ilk N çift (alt küme)."""
    scenes = []
    for name in MEGA1500_SCENES:
        z = np.load(f"{data_root}/{name}", allow_pickle=True)
        s = {k: z[k] for k in ("pair_infos", "intrinsics", "poses", "image_paths")}
        if limit is not None:
            s["pair_infos"] = s["pair_infos"][:limit]
        scenes.append(s)
    return scenes


def mega1500_eval(model, data_root: str, scenes: list[dict], c4: bool = False,
                  n_ransac_repeats: int = 5) -> dict:
    """romav2 Mega1500.benchmark ile AYNI döngü; c4=True → i. çiftin B'si (i mod 4)·90°.

    Kayıt: romav2 2.0.1 benchmarks/mega1500.py (1200 px K ölçeklemesi, 5 × 5000 eşleşme,
    eşik 0,5 px, conf 0,99999). c4=False iken sonuç orijinalle aynı olmalıdır (aynı tohum).
    """
    from PIL import Image
    from romav2.geometry import (
        compute_pose_error,
        compute_relative_pose,
        estimate_pose_cv2_ransac,
        pose_auc,
    )

    tot_e_pose, per_rot = [], {0: [], 1: [], 2: [], 3: []}
    i_global = 0
    for scene in scenes:
        pairs, intr, poses, paths = (scene[k] for k in ("pair_infos", "intrinsics", "poses",
                                                         "image_paths"))
        for pair in pairs:
            idx1, idx2 = pair[0]
            K1, T1 = intr[idx1].copy(), poses[idx1].copy()
            K2, T2 = intr[idx2].copy(), poses[idx2].copy()
            R1, t1, R2, t2 = T1[:3, :3], T1[:3, 3], T2[:3, :3], T2[:3, 3]
            im_A_path = f"{data_root}/{paths[idx1]}"
            im_B_path = f"{data_root}/{paths[idx2]}"
            im_A, im_B = Image.open(im_A_path), Image.open(im_B_path)
            w1, h1 = im_A.size
            w2, h2 = im_B.size
            k = (i_global % 4) if c4 else 0
            if k:
                K2, R2, t2, w2, h2 = rot90_camera(K2, R2, t2, w2, h2, k)
                B = np.ascontiguousarray(np.rot90(np.array(im_B.convert("RGB")), k))
                preds = model.match(np.array(im_A.convert("RGB")), B)
            else:
                preds = model.match(im_A_path, im_B_path)
            R, t = compute_relative_pose(R1, t1, R2, t2)
            scale1, scale2 = 1200 / max(w1, h1), 1200 / max(w2, h2)
            w1, h1, w2, h2 = scale1 * w1, scale1 * h1, scale2 * w2, scale2 * h2
            K1, K2 = K1.copy(), K2.copy()
            K1[:2] *= scale1
            K2[:2] *= scale2
            for _ in range(n_ransac_repeats):
                matches, _, _, _ = model.sample(preds, 5_000)
                kpts1, kpts2 = model.to_pixel_coordinates(matches, h1, w1, h2, w2)
                kpts1, kpts2 = kpts1.cpu().numpy(), kpts2.cpu().numpy()
                sh = np.random.permutation(np.arange(len(kpts1)))
                kpts1, kpts2 = kpts1[sh], kpts2[sh]
                try:
                    thr = 0.5 / (np.mean(np.abs(K1[:2, :2])) + np.mean(np.abs(K2[:2, :2])))
                    R_est, t_est, _ = estimate_pose_cv2_ransac(kpts1, kpts2, K1, K2, thr,
                                                               conf=0.99999)
                    e_t, e_R = compute_pose_error(R_est, t_est[:, 0], R, t)
                    e_pose = max(e_t, e_R)
                except Exception:  # noqa: BLE001 — orijinaldeki gibi: başarısız = 90°
                    e_pose = 90.0
                tot_e_pose.append(e_pose)
                per_rot[k].append(e_pose)
            i_global += 1
    thr = [5, 10, 20]
    out = {"n_pairs": i_global, "auc": [float(a) for a in pose_auc(np.array(tot_e_pose), thr)]}
    if c4:
        out["auc_by_rot"] = {f"{90 * k}": [float(a) for a in pose_auc(np.array(v), thr)]
                             for k, v in per_rot.items() if v}
    return out
