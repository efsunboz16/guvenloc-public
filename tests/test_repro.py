import numpy as np
import pytest

from guvenloc.repro import rot90_camera, rot90_points


def project(K, R, t, X):
    x = (K @ (R @ X.T + t[:, None])).T
    return x[:, :2] / x[:, 2:3]


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_rot90_camera_matches_image_rotation(k):
    rng = np.random.default_rng(k)
    W, H = 640.0, 480.0
    K = np.array([[500.0, 0, 330.0], [0, 520.0, 235.0], [0, 0, 1]])
    a = rng.normal(size=3) * 0.2
    R, _ = np.linalg.qr(np.eye(3) + np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]]))
    R *= np.sign(np.linalg.det(R))
    t = np.array([0.1, -0.2, 0.3])
    X = rng.uniform([-2, -2, 4], [2, 2, 8], (50, 3))
    X = (R.T @ (X.T - t[:, None])).T  # kamera önünde olsun diye dünya koordinatına çevir
    x = project(K, R, t, X)
    K2, R2, t2, W2, H2 = rot90_camera(K, R, t, W, H, k)
    np.testing.assert_allclose(project(K2, R2, t2, X), rot90_points(x, W, H, k), atol=1e-8)
    assert (W2, H2) == ((W, H) if k % 2 == 0 else (H, W))
    assert np.allclose(R2 @ R2.T, np.eye(3)) and np.isclose(np.linalg.det(R2), 1)


def test_rot90_points_matches_numpy():
    W, H = 5, 3
    img = np.arange(W * H).reshape(H, W)
    r = np.rot90(img, 1)
    centers = np.array([[x + 0.5, y + 0.5] for y in range(H) for x in range(W)])
    q = rot90_points(centers, W, H, 1) - 0.5
    np.testing.assert_array_equal(r[q[:, 1].round().astype(int), q[:, 0].round().astype(int)],
                                  img.ravel())


def test_pose_auc_runs_on_numpy2():
    """romav2.geometry.pose_auc np.trapz kullanır; guvenloc.repro uyum yamasıyla çalışmalı."""
    pytest.importorskip("romav2")  # CI'da GPU paketleri kurulmaz
    from romav2.geometry import pose_auc

    err = np.array([0.5, 1.0, 3.0, 7.0, 15.0, 30.0])
    auc = pose_auc(err, [5, 10, 20])
    # elle: ölçü eğrisinin trapez integrali
    ref = []
    for t in (5, 10, 20):
        e = np.sort(err)
        r = (np.arange(len(e)) + 1) / len(e)
        e, r = np.r_[0.0, e], np.r_[0.0, r]
        i = np.searchsorted(e, t)
        x, y = np.r_[e[:i], t], np.r_[r[:i], r[i - 1]]
        ref.append(np.trapezoid(y, x=x) / t)
    np.testing.assert_allclose(auc, ref, rtol=1e-12)
