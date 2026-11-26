"""Modül 2 v0 oylama hacmi — sentetik çiftlerde geometri ve metrikler (MIMARI 3A–3B)."""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # CI'da GPU paketleri kurulmaz

from guvenloc import m2_volume as V

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def synth(theta_deg=37.0, s=0.13, gt=(410.0, 530.0), n=3000, noise_px=0.5, outlier=0.0, seed=0, wh=(3976, 2652)):
    """Drone pikselleri u; karo v = gt + s R(θ)(u − u0) + gürültü (MIMARI 3A kuralının tersi)."""
    rng = np.random.default_rng(seed)
    u = rng.uniform([0, 0], wh, size=(n, 2)).astype(np.float32)
    u0 = np.array([(wh[0] - 1) / 2, (wh[1] - 1) / 2])
    th = math.radians(theta_deg)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    v = np.asarray(gt) + s * (u - u0) @ R.T + rng.normal(0, noise_px, (n, 2))
    k = int(outlier * n)
    v[:k] = rng.uniform(0, 1024, (k, 2))
    S = np.tile(np.eye(2, dtype=np.float32) * 1.0, (n, 1, 1))
    return u, v.astype(np.float32), np.ones(n, np.float32), S, np.zeros_like(S), s, wh


@pytest.mark.parametrize("theta", [0.0, 37.0, 181.5, 300.0])
def test_peak_recovers_pose(theta):
    u, v, c, SB, SA, s, wh = synth(theta_deg=theta)
    vol = V.build_volume(u, v, c, SB, SA, s, wh, 1024, 1.0, device=DEV)
    m = V.metrics(vol)
    assert math.hypot(m["x"] - 410, m["y"] - 530) < 2.0           # 2 m hücre, alt-hücre inceltmesi
    assert abs((m["theta"] - theta + 180) % 360 - 180) < 1.0
    assert m["rho"] < 0.05
    assert m["support_frac"] > 0.9


def test_m1_center_mode():
    u, v, c, SB, SA, s, wh = synth(theta_deg=95.0)
    m = V.metrics(V.build_volume(u, v, c, SB, SA, s, wh, 1024, 1.0, theta_center=90.0, device=DEV))
    assert abs(m["theta"] - 95.0) < 1.0


def test_negative_is_diffuse():
    rng = np.random.default_rng(1)
    u, _, c, SB, SA, s, wh = synth()
    v = rng.uniform(0, 1024, (len(u), 2)).astype(np.float32)
    neg = V.metrics(V.build_volume(u, v, c, SB, SA, s, wh, 1024, 1.0, device=DEV))
    pos = V.metrics(V.build_volume(*synth(outlier=0.7), 1024, 1.0, device=DEV))
    assert neg["support"] < 0.1 * pos["support"]
    assert neg["entropy_bits"] > pos["entropy_bits"]
    assert neg["rho"] > pos["rho"]


def test_splat_conserves_mass():
    mc = torch.tensor([[[20.3, 30.7], [50.0, 50.0]]], device=DEV)
    vol = V.splat(mc, torch.tensor([1.0, 2.0], device=DEV), torch.tensor([1.5, 3.0], device=DEV), 100, 100, 4)
    assert abs(float(vol.sum()) - 3.0) < 1e-3


def test_crop_centers_and_pads():
    vol = torch.arange(2 * 10 * 10, dtype=torch.float32).view(2, 10, 10)
    c = V.crop(vol, 1, 8, 3)
    assert c.shape == (2, 6, 6)
    assert float(c[0, 3, 3]) == float(vol[0, 1, 8])
    assert float(c[0, 0, 0]) == 0.0          # y < 0 dolgusu
