"""Modül 2 v0 — eğitimsiz oylama hacmi P(x, y, θ) ve hacim metrikleri (MIMARI.md 3A–3B).

Her eşleşme i (drone pikseli u_i ↔ karo pikseli v_i) ve yön hipotezi θ için drone merkezinin karodaki adayı:

    m_i(θ) = v_i − s · R(θ) · (u_i − u_0),     R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]]

θ = görüntü üst kenarının pusula yönü (kuzey 0, saat yönü; pairs `gt_heading` ile aynı kural); s = drone pikseli →
karo pikseli (`scale_s`); u_0 = drone karesinin merkezi (orijinal, döndürülmemiş kare; u önbellekte hep bu
koordinatta). İşaret gerçek çiftlerde RANSAC homografisiyle doğrulandı (inlier'larda medyan 3,4 px; ters işaret 76 px).

    P(x, y, θ) ∝ Σ_i c_i · N((x, y); m_i(θ), σ_i² I),   σ_i² = σ_0² + iz(Σ_i^map) / 2
    Σ_i^map = Σ_B,i + s² R Σ_A,i Rᵀ      (iz dönmeden bağımsız → σ_i θ'ya bağlı değil)

Yaklaşım (v0): kovaryans izotrop alınır (Σ_B'nin özdeğerleri iyi eşleşmelerde ≈ 1–3 m, σ_0 = 4 m'nin altında; plan
tam 2×2 Σ diyordu). Eşleşmeler σ_i'ye göre geometrik dilimlere ayrılır; her dilim çift doğrusal saçılır ve
dilimin σ'sıyla ayrılabilir Gauss ile bulanıklaştırılır (eşleşme başına ±3σ penceresine eşdeğer, çok daha hızlı).
σ_i > σ_cap olan eşleşmeler atılır (hacme katkıları zaten dağınık); oranı `frac_dropped`.

θ araması (karar F3-1): önce tam çember (5°, σ_0 kaba = 12 m: 2,5°'lik açı hatası karo kenarında ≈ 10 m kaydırır),
sonra kaba tepe çevresinde ±6° / 1° (σ_0 = 4 m). `theta_center` verilirse (M1) yalnız ±15° / 1° (plandaki kurulum).
Metrikler ince hacimden.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VolumeCfg:
    cell_m: float = 2.0
    sigma0_m: float = 4.0
    sigma0_coarse_m: float = 12.0
    coarse_step_deg: float = 5.0
    fine_half_deg: float = 6.0
    fine_step_deg: float = 1.0
    m1_half_deg: float = 15.0
    sigma_cap_m: float = 30.0
    n_bins: int = 6
    spread_radius_m: float = 50.0
    nms_radius_m: float = 50.0
    support_radius_m: float = 25.0
    sigma_scale: float = 1.0          # config.sigma_scale / meta.sigma_scale (INTERFACES §1)


def candidates(u: torch.Tensor, v: torch.Tensor, s: float, u0: torch.Tensor, thetas_deg: torch.Tensor) -> torch.Tensor:
    """(T, N, 2) karo pikseli: m_i(θ)."""
    th = torch.deg2rad(thetas_deg)[:, None]
    c, sn = torch.cos(th), torch.sin(th)
    d = u - u0
    rx = c * d[:, 0] - sn * d[:, 1]
    ry = sn * d[:, 0] + c * d[:, 1]
    return torch.stack([v[:, 0] - s * rx, v[:, 1] - s * ry], -1)


def _gauss1d(sigma: float, dev) -> torch.Tensor:
    r = max(1, math.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _blur(vol: torch.Tensor, sigma: float) -> torch.Tensor:
    """(T, Y, X) ayrılabilir Gauss; kütleyi korur (kenarda taşan kısım düşer)."""
    k = _gauss1d(sigma, vol.device)
    r = (len(k) - 1) // 2
    x = vol[:, None]
    x = F.conv2d(x, k.view(1, 1, 1, -1), padding=(0, r))
    x = F.conv2d(x, k.view(1, 1, -1, 1), padding=(r, 0))
    return x[:, 0]


def splat(mc: torch.Tensor, w: torch.Tensor, sig: torch.Tensor, ny: int, nx: int, n_bins: int) -> torch.Tensor:
    """mc (T, N, 2) hücre koordinatı (x, y; hücre merkezi tamsayı), w (N,), sig (N,) hücre → (T, ny, nx)."""
    T = mc.shape[0]
    vol = torch.zeros(T, ny, nx, device=mc.device)
    if len(w) == 0:
        return vol
    lo, hi = float(sig.min()), float(sig.max())
    edges = torch.logspace(math.log10(lo), math.log10(hi * 1.0001 + 1e-6), n_bins + 1, device=mc.device) \
        if hi > lo * 1.05 else torch.tensor([lo * 0.999, hi * 1.001 + 1e-6], device=mc.device)
    b = torch.bucketize(sig, edges[1:-1])
    t_idx = torch.arange(T, device=mc.device)[:, None]
    for k in range(len(edges) - 1):
        sel = b == k
        if not sel.any():
            continue
        acc = torch.zeros(T * ny * nx, device=mc.device)
        p = mc[:, sel]
        ww = w[sel][None].expand(T, -1)
        x0, y0 = torch.floor(p[..., 0]), torch.floor(p[..., 1])
        fx, fy = p[..., 0] - x0, p[..., 1] - y0
        for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = (x0 + dx).long(), (y0 + dy).long()
            ok = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
            idx = (t_idx * ny + yi) * nx + xi
            acc.index_add_(0, idx[ok], (ww * wt)[ok])
        s_b = float(torch.sqrt(edges[k] * edges[k + 1]))       # dilimin geometrik ortası
        vol += _blur(acc.view(T, ny, nx), s_b)
    return vol


@torch.no_grad()
def build_volume(u, v, c, Sigma_B, Sigma_A, scale_s: float, drone_wh, tile_px: int, mpp: float,
                 cfg: VolumeCfg = VolumeCfg(), theta_center: float | None = None, device="cuda",
                 density: bool = False) -> dict:
    """Ham (normalize edilmemiş) ince hacim + ızgara bilgisi. Girdi numpy (önbellek alanları).
    density=True: aynı ızgarada ağırlıksız (c = 1) oy hacmi de döner ("dens"; 3D inceltici girdisi)."""
    dev = torch.device(device)
    u_t = torch.as_tensor(np.asarray(u, np.float32), device=dev)
    v_t = torch.as_tensor(np.asarray(v, np.float32), device=dev)
    c_t = torch.as_tensor(np.asarray(c, np.float32), device=dev)
    trB = torch.as_tensor(np.trace(np.asarray(Sigma_B, np.float32), axis1=1, axis2=2), device=dev)
    trA = torch.as_tensor(np.trace(np.asarray(Sigma_A, np.float32), axis1=1, axis2=2), device=dev)
    var_px = cfg.sigma_scale ** 2 * (trB + scale_s ** 2 * trA) / 2         # karo pikseli²
    cell_px = cfg.cell_m / mpp
    n = math.ceil(tile_px / cell_px)
    u0 = torch.tensor([(drone_wh[0] - 1) / 2, (drone_wh[1] - 1) / 2], device=dev)

    def vol_for(thetas, sigma0_m, w=None):
        w = c_t if w is None else w
        sig = torch.sqrt(var_px / cell_px ** 2 + (sigma0_m / cfg.cell_m) ** 2)
        keep = sig <= cfg.sigma_cap_m / cfg.cell_m
        # karo pikseli p (merkez tamsayı) → hücre koordinatı: hücre j = [j·cell_px − ½, (j+1)·cell_px − ½)
        mc = (candidates(u_t[keep], v_t[keep], scale_s, u0, thetas) + 0.5) / cell_px - 0.5
        return splat(mc, w[keep], sig[keep], n, n, cfg.n_bins), float(1 - keep.float().mean())

    if theta_center is None:
        th_c = torch.arange(0, 360, cfg.coarse_step_deg, device=dev)
        vc, _ = vol_for(th_c, cfg.sigma0_coarse_m)
        t0 = float(th_c[int(vc.flatten(1).amax(1).argmax())])
        half = cfg.fine_half_deg
    else:
        t0, half = float(theta_center), cfg.m1_half_deg
    th_f = t0 + torch.arange(-half, half + 1e-6, cfg.fine_step_deg, device=dev)
    vf, dropped = vol_for(th_f, cfg.sigma0_m)
    out = {"vol": vf, "thetas": th_f % 360, "cell_px": cell_px, "n": n, "mpp": mpp, "cfg": cfg,
           "sum_c": float(c_t.sum()), "n_matches": len(c_t), "frac_dropped": dropped}
    if density:
        out["dens"] = vol_for(th_f, cfg.sigma0_m, torch.ones_like(c_t))[0]
    return out


def crop(vol: torch.Tensor, cy: int, cx: int, half: int) -> torch.Tensor:
    """(T, Y, X) → (T, 2·half, 2·half), merkez hücre (cy, cx) = indeks (half, half); taşan kısım 0."""
    T, ny, nx = vol.shape
    out = vol.new_zeros(T, 2 * half, 2 * half)
    y0, x0 = cy - half, cx - half
    ys, xs = max(0, y0), max(0, x0)
    ye, xe = min(ny, y0 + 2 * half), min(nx, x0 + 2 * half)
    if ye > ys and xe > xs:
        out[:, ys - y0:ye - y0, xs - x0:xe - x0] = vol[:, ys:ye, xs:xe]
    return out


def _quad(a_m, a_0, a_p) -> float:
    den = a_m - 2 * a_0 + a_p
    return float(0.5 * (a_m - a_p) / den) if den < 0 else 0.0


@torch.no_grad()
def metrics(V: dict) -> dict:
    """3B: tepe (x̂, ŷ karo pikseli; θ̂), yayılım σ̂ (m, tepe diliminde), tepe oranı ρ (θ üzerinden toplanmış
    P_xy'de, tepeden nms_radius dışındaki en büyük / tepe), entropi H (bit, tüm hacim), destek (tepe dilimde
    support_radius içindeki ham kütle ≈ "ağırlıklı inlier"), kütleler."""
    vol, cfg, cp = V["vol"], V["cfg"], V["cell_px"]
    T, n, _ = vol.shape
    tot = float(vol.sum())
    out = {"sum_c": V["sum_c"], "n_matches": V["n_matches"], "frac_dropped": V["frac_dropped"], "mass": tot}
    if tot <= 0:
        return out | {"x": np.nan, "y": np.nan, "theta": np.nan, "spread_m": np.nan, "rho": 1.0,
                      "entropy_bits": np.nan, "peak_p": 0.0, "support": 0.0, "support_frac": 0.0}
    P = vol / tot
    k = int(P.argmax())
    t, yx = divmod(k, n * n)
    y, x = divmod(yx, n)
    sl = P[t]
    dx = _quad(sl[y, x - 1], sl[y, x], sl[y, x + 1]) if 0 < x < n - 1 else 0.0
    dy = _quad(sl[y - 1, x], sl[y, x], sl[y + 1, x]) if 0 < y < n - 1 else 0.0
    dt = _quad(P[t - 1, y, x], P[t, y, x], P[t + 1, y, x]) if 0 < t < T - 1 else 0.0
    step = float(V["thetas"][1] - V["thetas"][0]) % 360 if T > 1 else 0.0
    xc, yc = x + dx, y + dy
    Pxy = P.sum(0)
    yy, xx = torch.meshgrid(torch.arange(n, device=vol.device), torch.arange(n, device=vol.device), indexing="ij")
    d2 = ((xx - xc) ** 2 + (yy - yc) ** 2) * cfg.cell_m ** 2
    m = d2 <= cfg.spread_radius_m ** 2
    w = sl * m                                    # yayılım tepe θ diliminde (θ toplamı yanlış θ'larla şişirir)
    ws = w.sum()
    mx, my = (w * xx).sum() / ws, (w * yy).sum() / ws
    cxx = (w * (xx - mx) ** 2).sum() / ws
    cyy = (w * (yy - my) ** 2).sum() / ws
    cxy = (w * (xx - mx) * (yy - my)).sum() / ws
    ev = torch.linalg.eigvalsh(torch.stack([torch.stack([cxx, cxy]), torch.stack([cxy, cyy])]))
    spread = float(torch.sqrt(ev.clamp(min=0).mean())) * cfg.cell_m
    outside = Pxy.masked_fill(d2 <= cfg.nms_radius_m ** 2, 0)
    rho = float(outside.max() / Pxy.max())
    pf = P[P > 0]
    H = float(-(pf * torch.log2(pf)).sum())
    sup = float((vol[t] * (d2 <= cfg.support_radius_m ** 2)).sum())
    return out | {"x": (xc + 0.5) * cp - 0.5, "y": (yc + 0.5) * cp - 0.5,
                  "theta": float(V["thetas"][t] + dt * step) % 360, "spread_m": spread, "rho": rho,
                  "entropy_bits": H, "peak_p": float(P.max()), "support": sup,
                  "support_frac": sup / max(V["sum_c"], 1e-9)}
