"""scripts/13_roma_heading_scale.py geometri yardımcıları (GPU gerektirmez)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("pandas")
pytest.importorskip("torch")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "13_roma_heading_scale.py"
spec = importlib.util.spec_from_file_location("rhs", SCRIPT)
rhs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rhs)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_unrot90_roundtrip(k):
    h, w = 5, 7
    img = np.arange(h * w).reshape(h, w)
    r = rhs.rot90_img(img, k)
    ys, xs = np.mgrid[: r.shape[0], : r.shape[1]]
    pts = np.stack([xs.ravel(), ys.ravel()], 1).astype(float)
    orig = rhs.unrot90_pts(pts, k, w, h).round().astype(int)
    np.testing.assert_array_equal(img[orig[:, 1], orig[:, 0]], r.ravel())


@pytest.mark.parametrize("bearing", [0, 30, 90, 200, 315])
def test_similarity_bearing(bearing):
    # Görüntü üstü kuzeyden saat yönünde `bearing` derece bakıyorsa, drone→harita dönüşümü
    # görüntüyü saat yönünde (y aşağı koordinatta +bearing) döndürür.
    t = np.radians(bearing)
    M = 2.0 * np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0]])
    s, b = rhs.similarity_params(M)
    assert s == pytest.approx(2.0)
    assert b == pytest.approx(bearing % 360)
    up = M[:, :2] @ np.array([0, -1.0])                  # görüntü yukarısı haritada nereye?
    east, north = up[0], -up[1]
    assert np.degrees(np.arctan2(east, north)) % 360 == pytest.approx(bearing % 360)
