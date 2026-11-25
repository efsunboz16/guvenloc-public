import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from guvenloc.matcher import (
    MatchSet,
    precision_grid_to_image,
    rotate_image,
    rotation_affine,
    unrotate_cov,
    unrotate_points,
)


@pytest.mark.parametrize("rot", [0, 90, 180, 270, -90, 450])
def test_rot90_affine_matches_pixels(rot):
    h, w = 5, 7
    img = np.arange(h * w).reshape(h, w)
    r, M = rotate_image(img, rot)
    ys, xs = np.mgrid[:h, :w]
    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(float)
    q = (pts @ M[:, :2].T + M[:, 2]).round().astype(int)
    np.testing.assert_array_equal(r[q[:, 1], q[:, 0]], img.ravel())
    np.testing.assert_allclose(unrotate_points(q.astype(float), M), pts, atol=1e-9)


def test_rotation_is_clockwise():
    # Saat yönünde 90°: görüntünün üst kenarı sağa gider
    M, (W2, H2) = rotation_affine(10, 4, 90)
    top_center = np.array([4.5, 0.0])
    p = M[:, :2] @ top_center + M[:, 2]
    assert (W2, H2) == (4, 10) and p[0] == pytest.approx(3.0) and p[1] == pytest.approx(4.5)


def test_arbitrary_angle_roundtrip_and_canvas():
    img = np.zeros((40, 60, 3), np.uint8)
    img[10, 20] = 255
    r, M = rotate_image(img, 30)
    assert r.shape[1] >= 60 and r.shape[0] >= 40
    y, x = np.unravel_index(r[..., 0].argmax(), r.shape[:2])
    back = unrotate_points(np.array([[x, y]], float), M)[0]
    assert np.allclose(back, [20, 10], atol=1.0)
    S = np.diag([4.0, 1.0])[None]
    S2 = unrotate_cov(np.einsum("ij,njk,lk->nil", M[:, :2], S, M[:, :2]), M)
    np.testing.assert_allclose(S2, S, atol=1e-9)


def test_precision_grid_to_image():
    P = np.eye(2)[None] * 4.0                      # σ = 0,5 ızgara pikseli
    Pi = precision_grid_to_image(P, (640, 640), (1280, 320))
    S = np.linalg.inv(Pi[0])
    assert np.sqrt(S[0, 0]) == pytest.approx(1.0)   # x: 2× büyütme → σ 1 px
    assert np.sqrt(S[1, 1]) == pytest.approx(0.25)  # y: 0,5× → σ 0,25 px


def test_matchset_npz_roundtrip(tmp_path):
    n = 10
    ms = MatchSet(np.random.rand(n, 2).astype(np.float32) * 100,
                  np.random.rand(n, 2).astype(np.float32) * 100,
                  np.random.rand(n).astype(np.float32),
                  np.tile(np.eye(2, dtype=np.float32), (n, 1, 1)) * 0.3,
                  np.tile(np.eye(2, dtype=np.float32), (n, 1, 1)) * 2.0, {"rot_deg": 90.0})
    p = tmp_path / "a" / "x.npz"
    ms.save_npz(p)
    back = MatchSet.load_npz(p)
    np.testing.assert_allclose(back.u, ms.u)
    np.testing.assert_allclose(back.Sigma_B, ms.Sigma_B, rtol=1e-3)
    assert back.meta["rot_deg"] == 90.0 and "sigma_eig_range_px2" in back.meta and len(back) == n
    assert not list(p.parent.glob("*.tmp.npz"))


@pytest.mark.gpu
@pytest.mark.parametrize("rot", [0, 90, 180, 270, 37])
def test_matcher_synthetic_homography(rot):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("GPU yok")
    import matplotlib

    from guvenloc.matcher import Matcher

    src = cv2.imread(str(__import__("pathlib").Path(matplotlib.get_data_path())
                         / "sample_data" / "grace_hopper.jpg"))
    A = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    h, w = A.shape[:2]
    ang = np.radians(20)
    H = np.eye(3)
    H[:2, :2] = 0.9 * np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    H[:2, 2] = [w / 2, h / 2] - H[:2, :2] @ [w / 2, h / 2] + [10, -5]
    B = cv2.warpPerspective(A, H, (w, h))
    m = _matcher(Matcher)
    ms = m.match(A, B, rot_deg=rot, n=3000)
    pred = cv2.perspectiveTransform(ms.u[None].astype(np.float64), H)[0]
    err = np.linalg.norm(pred - ms.v, axis=1)
    inside = (pred[:, 0] > 5) & (pred[:, 0] < w - 5) & (pred[:, 1] > 5) & (pred[:, 1] < h - 5)
    assert np.median(err[inside & (ms.c > 0.5)]) < 1.0
    sig = np.sqrt(np.trace(ms.Sigma_B, axis1=1, axis2=2) / 2)
    assert 0.01 < np.median(sig) < 5.0             # piksel mertebesi (normalize değil)
    assert ms.meta["rot_deg"] == rot


_M = {}


def _matcher(cls):
    if "m" not in _M:
        _M["m"] = cls(setting="base")
    return _M["m"]


def test_save_clamps_huge_and_degenerate_cov(tmp_path):
    from guvenloc.cache import validate_npz
    from guvenloc.matcher import SIGMA_EIG_RANGE_PX2

    n = 3
    SB = np.array([np.diag([1e7, 1.0]), np.diag([1e-9, 4.0]), np.eye(2)], np.float32)
    ms = MatchSet(np.full((n, 2), 5, np.float32), np.full((n, 2), 5, np.float32),
                  np.full(n, 0.5, np.float32), np.zeros((n, 2, 2), np.float32), SB,
                  {"setting": "base", "romav2": "2.0.1", "rot_deg": 0.0, "drone_wh": [10, 10],
                   "sat_wh": [10, 10], "grid_wh": [640, 640], "sigma_scale": 1.0,
                   "bidirectional": False, "coords": "opencv", "cov_units": "px2"})
    p = tmp_path / "x.npz"
    ms.save_npz(p)
    assert validate_npz(p) == []
    back = MatchSet.load_npz(p)
    assert back.Sigma_B[0, 0, 0] == pytest.approx(SIGMA_EIG_RANGE_PX2[1], rel=1e-3)
    assert (back.Sigma_A == 0).all()
