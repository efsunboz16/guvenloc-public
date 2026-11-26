"""Modül 1 — kuzey kanonikleştirici (MIMARI Faz 2B).

Drone karesinden görüntü üst kenarının pusula yönünü φ (saat yönü, derece; φ = 0 → kuzey-yukarı) tahmin eder.
Kare φ̂ kadar SAAT YÖNÜNDE döndürülünce kuzey-yukarı olur (`geo.north_up_rotation_cw_deg` ile aynı kural).

Ağ: escnn C8 (rot2dOnR2(N=8)) eşdeğer CNN. Son katman tek bir regular alan verir: 8 yön için 8 logit
(n·45°). 36 bölmeli logit, bu 8 değerin bant sınırlı (Fourier, frekans ≤ 4) interpolasyonudur → girdi 45°·k
döndürülünce 36 bölmeli çıktı tam 4k bölme kayar (piksel ızgarası için 90°'nin katlarında sayısal olarak tam,
45°'nin tek katlarında ve ara açılarda yaklaşık — `tests/test_m1_equivariance.py`).

Dönme kuralı: içerik saat yönünde α döndürülürse φ' = φ − α (mod 360).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N_BINS = 36
BIN_DEG = 360.0 / N_BINS
N_GROUP = 8
# Girdi kenarı TEK sayı olmalı (2^n + 1): adımlı katmanların örnekleme ızgarası ancak o zaman merkeze göre
# simetrik kalır ve 90°'lik dönmelerde eşdeğerlik tam olur (256'da bozuluyor — ölçüldü).
IN_SIZE = 257


# ---------------------------------------------------------------------------
# Girdi hazırlığı
# ---------------------------------------------------------------------------

def prep_square(img_bgr: np.ndarray, size: int = 321) -> np.ndarray:
    """Merkez kare kırpma (kısa kenar) + yeniden boyutlandırma → (size, size, 3) uint8 RGB.

    Dairesel maske sonra uygulanır: karenin iç çemberi her dönmede görüntü içinde kalır.
    """
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    sq = img_bgr[y0:y0 + s, x0:x0 + s]
    sq = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(sq, cv2.COLOR_BGR2RGB)


def circular_mask(size: int, device=None) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device),
                            indexing="ij")
    c = (size - 1) / 2
    return (((yy - c) ** 2 + (xx - c) ** 2) <= (size / 2) ** 2).float()


def rotate_cw(x: torch.Tensor, angle_cw_deg: torch.Tensor, out_size: int | None = None,
              zoom: torch.Tensor | None = None) -> torch.Tensor:
    """(B,C,H,W) görüntüleri merkez etrafında SAAT YÖNÜNDE döndürür (bilineer), isteğe bağlı yeniden
    boyutlandırır ve dairesel maske uygular. zoom > 1 içeriği büyütür (ölçek oynatma)."""
    B, _, H, _ = x.shape
    out = out_size or H
    a = torch.deg2rad(angle_cw_deg.to(x.dtype)).view(B)
    c, s = torch.cos(a), torch.sin(a)
    if zoom is not None:
        z = zoom.to(x.dtype).view(B)
        c, s = c / z, s / z
    # affine_grid çıktı noktasını girdiye eşler (x sağ, y aşağı): içeriği saat yönünde α döndürmek için
    # girdi = R(−α)·çıktı (rotate_cw(·, 90) == rot90 saat yönü, testli)
    theta = torch.stack([torch.stack([c, s, torch.zeros_like(c)], 1),
                         torch.stack([-s, c, torch.zeros_like(c)], 1)], 1)
    grid = F.affine_grid(theta, (B, x.shape[1], out, out), align_corners=False)
    y = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return y * circular_mask(out, x.device)


def normalize(x_uint8: torch.Tensor) -> torch.Tensor:
    """(B,H,W,3) uint8 → (B,3,H,W) float, ImageNet istatistikleriyle."""
    x = x_uint8.permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Etiket ve çıktı yardımcıları
# ---------------------------------------------------------------------------

def bin_centers_deg() -> np.ndarray:
    return (np.arange(N_BINS) + 0.5) * BIN_DEG


def soft_targets(phi_deg: torch.Tensor, sigma_bins: float = 1.0) -> torch.Tensor:
    """Dairesel Gauss etiket yumuşatma: (B,) derece → (B,36) olasılık."""
    centers = torch.tensor(bin_centers_deg(), device=phi_deg.device, dtype=torch.float32)
    d = torch.remainder(centers[None] - phi_deg[:, None].float() + 180.0, 360.0) - 180.0
    w = torch.exp(-0.5 * (d / (sigma_bins * BIN_DEG)) ** 2)
    return w / w.sum(1, keepdim=True)


def fourier_interp_matrix(n_out: int = N_BINS, n_in: int = N_GROUP, sign: int = 1) -> torch.Tensor:
    """8 eşit aralıklı örnekten n_out açıya bant sınırlı (Dirichlet) interpolasyon, (n_out, n_in).

    Çıkış açısı θ_j = bölme merkezi; giriş n, θ_n = sign·n·45° yönüne karşılık gelir.
    """
    th_out = np.deg2rad(bin_centers_deg())[:, None]
    th_in = np.deg2rad(sign * np.arange(n_in) * 360.0 / n_in)[None, :]
    d = th_out - th_in
    m = np.ones_like(d)
    for k in range(1, n_in // 2):
        m += 2 * np.cos(k * d)
    m += np.cos((n_in // 2) * d)  # Nyquist terimi (çift n_in)
    return torch.tensor(m / n_in, dtype=torch.float32)


def predict_deg(logits: torch.Tensor) -> torch.Tensor:
    """argmax bölmesi + komşularla parabolik alt-bölme inceltmesi → (B,) derece [0, 360)."""
    k = logits.argmax(1)
    idx = torch.arange(len(k), device=logits.device)
    lm, l0, lp = logits[idx, (k - 1) % N_BINS], logits[idx, k], logits[idx, (k + 1) % N_BINS]
    den = lm - 2 * l0 + lp
    off = torch.where(den.abs() > 1e-9, 0.5 * (lm - lp) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)
    return torch.remainder((k.float() + 0.5 + off) * BIN_DEG, 360.0)


def angular_error_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (torch.remainder(a - b + 180.0, 360.0) - 180.0).abs()


# ---------------------------------------------------------------------------
# Ağ
# ---------------------------------------------------------------------------

# 8 yönlü regular alanın n. bileşeni hangi yöne karşılık geliyor: escnn'de grup elemanı n, n·45° saat
# YÖNÜNÜN TERSİ dönmedir; içerik saat tersine dönünce φ artar (φ' = φ + α) → sign = +1.
# tests/test_m1_equivariance.py bu işareti sayısal olarak sınar.
REGULAR_SIGN = 1


class C8Canon(nn.Module):
    """C8 eşdeğer yön tahmincisi. Girdi (B,3,S,S) normalize, dairesel maskeli; çıktı (B,36) logit."""

    def __init__(self, widths: tuple[int, ...] = (16, 32, 64, 96, 128), in_size: int = IN_SIZE):
        super().__init__()
        from escnn import gspaces
        from escnn import nn as enn

        self.enn = enn
        g = gspaces.rot2dOnR2(N=N_GROUP)
        self.in_type = enn.FieldType(g, 3 * [g.trivial_repr])
        layers = []
        t_in = self.in_type
        for i, w in enumerate(widths):
            t_out = enn.FieldType(g, w * [g.regular_repr])
            k, stride = (7, 2) if i == 0 else (3, 1)
            layers += [enn.R2Conv(t_in, t_out, kernel_size=k, padding=k // 2, stride=stride, bias=False),
                       enn.InnerBatchNorm(t_out), enn.ReLU(t_out, inplace=True)]
            if i > 0:
                layers.append(enn.PointwiseAvgPoolAntialiased(t_out, sigma=0.66, stride=2))
            t_in = t_out
        self.body = enn.SequentialModule(*layers)
        self.head = enn.R2Conv(t_in, enn.FieldType(g, [g.regular_repr]), kernel_size=1)
        self.register_buffer("interp", fourier_interp_matrix(sign=REGULAR_SIGN))
        self.log_beta = nn.Parameter(torch.zeros(()))   # logit ölçeği (keskinlik)
        self.in_size = in_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(self.enn.GeometricTensor(x, self.in_type))
        y = self.head(y).tensor                      # (B, 8, h, w)
        v = y.mean(dim=(2, 3))                       # uzaysal ortalama: dönmeye değişmez, 8 yön logiti
        return torch.exp(self.log_beta) * v @ self.interp.T   # (B, 36)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def rotate_cw_90k(x: torch.Tensor, k: int) -> torch.Tensor:
    """Tam ızgara dönmesi: saat yönünde k·90° (test ve önbellek yönleriyle uyumlu)."""
    return torch.rot90(x, -k, dims=(2, 3))


def heading_after_cw(phi_deg, alpha_cw_deg):
    """İçerik saat yönünde α döndürülünce yeni yön: φ − α (mod 360)."""
    return (np.asarray(phi_deg) - np.asarray(alpha_cw_deg)) % 360.0


def to_bins(phi_deg) -> np.ndarray:
    return (np.floor(np.asarray(phi_deg) / BIN_DEG).astype(int)) % N_BINS



# ---------------------------------------------------------------------------
# Öneri C (fizibilite): drone karesi + kuzey-yukarı uydu karosu → göreli dönme
# ---------------------------------------------------------------------------

class C8RelRot(nn.Module):
    """Siyam C8 eşdeğer gövde (ağırlıklar paylaşılır) + grup korelasyonu.

    Her görüntü için D regular alan, uzaysal ortalama → (B, D, 8) betimleyici. Uydu karosu kuzey-yukarı;
    drone betimleyicisi 8 grup kayması üzerinden uyduyla ilişkilendirilir: c_k = <roll(d_drone, ROLL_SIGN·k),
    d_uydu>. c_k, k·45° yönü için logit → 36 bölmeye Fourier interpolasyonu (C8Canon ile aynı çıktı).
    Drone girdisi 45°·k döndürülünce çıktı tam 4,5k bölme kayar; uydu girdisi döndürülmez.

    sat_pool = "mean": uydu betimleyicisi de küresel ortalama (karo ≈ drone ayak izi iken).
    sat_pool = "window": uydu öznitelik haritası `window` hücrelik kayan pencereyle ortalanır, her konumda
    korelasyon alınır, konumlar üzerinde yumuşak maksimum (logsumexp/γ) → karo drone ayak izinden çok büyükken
    (gerçek karo ≈ 3×) drone karesinin karodaki yerini model kendisi arar.
    """

    ROLL_SIGN = -1   # tests/test_m1_equivariance.py::test_relrot_sign sayısal olarak sınar

    def __init__(self, widths: tuple[int, ...] = (16, 32, 64, 96, 128), n_desc: int = 32,
                 sat_pool: str = "mean", window: int = 5):
        super().__init__()
        self.sat_pool, self.window = sat_pool, window
        from escnn import gspaces
        from escnn import nn as enn

        self.enn = enn
        g = gspaces.rot2dOnR2(N=N_GROUP)
        self.in_type = enn.FieldType(g, 3 * [g.trivial_repr])
        layers, t_in = [], self.in_type
        for i, w in enumerate(widths):
            t_out = enn.FieldType(g, w * [g.regular_repr])
            k, stride = (7, 2) if i == 0 else (3, 1)
            layers += [enn.R2Conv(t_in, t_out, kernel_size=k, padding=k // 2, stride=stride, bias=False),
                       enn.InnerBatchNorm(t_out), enn.ReLU(t_out, inplace=True)]
            if i > 0:
                layers.append(enn.PointwiseAvgPoolAntialiased(t_out, sigma=0.66, stride=2))
            t_in = t_out
        self.body = enn.SequentialModule(*layers)
        self.head = enn.R2Conv(t_in, enn.FieldType(g, n_desc * [g.regular_repr]), kernel_size=1)
        self.n_desc = n_desc
        self.register_buffer("interp", fourier_interp_matrix(sign=REGULAR_SIGN))
        self.log_beta = nn.Parameter(torch.tensor(2.3))   # kosinüs benzerliği ölçeği (≈ 10)
        self.log_gamma = nn.Parameter(torch.tensor(2.3))  # konum yumuşak maksimumunun keskinliği (window)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.enn.GeometricTensor(x, self.in_type))).tensor   # (B, D·8, h, w)

    def describe(self, x: torch.Tensor) -> torch.Tensor:
        d = self._features(x).mean(dim=(2, 3)).view(len(x), self.n_desc, N_GROUP)
        return F.normalize(d.flatten(1), dim=1).view_as(d)

    def describe_sat(self, x: torch.Tensor) -> torch.Tensor:
        """(B, D, 8) [mean] ya da (B, D, 8, P) [window] — açıdan bağımsız, değerlendirmede önbelleğe alınabilir."""
        if self.sat_pool == "mean":
            return self.describe(x)
        y = F.avg_pool2d(self._features(x), self.window, stride=1)
        d = y.flatten(2).view(len(x), self.n_desc, N_GROUP, -1)
        return d / d.flatten(1, 2).norm(dim=1).clamp_min(1e-6)[:, None, None]

    def score(self, dd: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
        rolls = torch.stack([dd.roll(self.ROLL_SIGN * k, dims=2) for k in range(N_GROUP)], 1)  # (B, 8, D, 8)
        if ds.dim() == 3:
            c = (rolls * ds[:, None]).sum((2, 3))
        else:
            cp = torch.einsum("bkdn,bdnp->bkp", rolls, ds)
            g = torch.exp(self.log_gamma)
            c = (torch.logsumexp(g * cp, dim=2) - math.log(cp.shape[2])) / g
        return torch.exp(self.log_beta) * c @ self.interp.T   # (B, 36)

    def forward(self, drone: torch.Tensor, sat: torch.Tensor) -> torch.Tensor:
        return self.score(self.describe(drone), self.describe_sat(sat))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
