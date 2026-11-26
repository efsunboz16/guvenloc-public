"""Modül 4 – zamansal füzyon: parçacık filtresi (MIMARI.md Faz 5B).

Durum (doğu, kuzey, θ): metre (uçuşa ait yerel ENU, eşdikdörtgen) ve görüntü üst kenarının pusula yönü (°, saat yönü;
M2 hacmiyle aynı kural). N parçacık.

  başlatma   ilk KABUL edilen tek kare kararında: konum yarıçapı q̂·σ̂ olan diskte düzgün, θ ~ N(θ̂, σ_θ0)
  tahmin     drone↔drone homografisinden drone çerçevesinde (ileri, sağ) m ve Δθ:
               (de, dn) = ileri · (sin θ, cos θ) + sağ · (cos θ, −sin θ)   (parçacığın θ'sıyla)
             gürültü: N(0, (σ₀ + r · |adım|)²) konumda, N(0, σ_θ²) yönde; p_jump olasılıkla ek N(0, σ_jump²) (ağır kuyruk)
  güncelleme yalnız M3 KABUL ettiyse: w ← w · ((P(x, y) / max P)^λ + ε) · ((1 − p) N(θ − θ̂; σ_θm) + p)
             P(x, y): θ üzerinden toplanmış M2 hacmi (karo ızgarası), parçacık konumunda çift doğrusal; dışarıda 0 → ε
             (sıfır ağırlık çöküşünü önler). Ret → güncelleme atlanır.
  sıfırlama  kabul edilmiş ölçüm bulutu açıklamıyorsa (ağırlıklı ortalama P/maxP < reset_lik) filtre o tek kare
             kararından yeniden başlatılır (sensör sıfırlaması; dönüşte kopan bulut karonun dışına düşer, ε tabanı onu geri
             getiremez).
  yeniden örnekleme  N_eff < oran · N ise sistematik.
  çıktı      ağırlıklı ortalama konum, dairesel ortalama θ; yayılım = ortalamaya uzaklığın ağırlıklı %90 yüzdeliği.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EARTH_R = 6_371_008.8


@dataclass
class FilterCfg:
    n_particles: int = 2000
    sigma_pos0_m: float = 5.0
    sigma_pos_rate: float = 0.2         # × adım uzunluğu
    sigma_theta_deg: float = 1.0
    jump_p: float = 0.02
    jump_sigma_m: float = 100.0
    lik_power: float = 1.0
    eps: float = 1e-3
    theta_meas_sigma_deg: float = 5.0   # ≥ 1e6 → θ ölçümü kapalı
    theta_outlier_p: float = 0.3         # θ̂ karışımı: (1 − p) N(θ − θ̂; σ) + p (yanlış tepede θ̂ anlamsız)
    theta_init_sigma_deg: float = 5.0
    resample_frac: float = 0.5
    reset_lik: float = 1e-3              # kabul edilmiş ölçümde ağırlıklı ortalama P/maxP bunun altındaysa → yeniden başlat
    seed: int = 0


def enu(lat0: float, lon0: float, lat, lon):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return (np.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0)), np.radians(lat - lat0) * EARTH_R)


@dataclass
class GridMeas:
    """Bir karenin P(x, y) ölçümü: karo merkezi `center_en` (m), karo pikseli mpp, hücre = cell_px piksel."""
    P: np.ndarray            # (n, n), toplam 1
    center_en: tuple[float, float]
    tile_px: int
    mpp: float
    cell_m: float

    def to_cell(self, e, n):
        x_px = self.tile_px / 2 - 0.5 + (np.asarray(e) - self.center_en[0]) / self.mpp
        y_px = self.tile_px / 2 - 0.5 - (np.asarray(n) - self.center_en[1]) / self.mpp
        cp = self.cell_m / self.mpp
        return (x_px + 0.5) / cp - 0.5, (y_px + 0.5) / cp - 0.5

    def to_en(self, x_px: float, y_px: float):
        return (self.center_en[0] + (x_px + 0.5 - self.tile_px / 2) * self.mpp,
                self.center_en[1] - (y_px + 0.5 - self.tile_px / 2) * self.mpp)

    def lookup(self, e, n) -> np.ndarray:
        """Çift doğrusal P / max P; ızgara dışı 0."""
        cx, cy = self.to_cell(e, n)
        P = self.P.astype(np.float32)
        pm = float(P.max())
        if pm <= 0:
            return np.zeros(len(cx))
        ny, nx = P.shape
        x0, y0 = np.floor(cx).astype(int), np.floor(cy).astype(int)
        fx, fy = cx - x0, cy - y0
        out = np.zeros(len(cx))
        for dy, wy in ((0, 1 - fy), (1, fy)):
            for dx, wx in ((0, 1 - fx), (1, fx)):
                xi, yi = x0 + dx, y0 + dy
                ok = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
                out[ok] += wx[ok] * wy[ok] * P[yi[ok], xi[ok]]
        return out / pm


def wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][min(np.searchsorted(cw, q * cw[-1]), len(x) - 1)])


class ParticleFilter:
    def __init__(self, cfg: FilterCfg | None = None):
        cfg = cfg or FilterCfg()
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.x = None                     # (N, 3): e, n, θ
        self.w = None

    @property
    def started(self) -> bool:
        return self.x is not None

    def init(self, e: float, n: float, radius_m: float, theta_deg: float) -> None:
        N, rng = self.cfg.n_particles, self.rng
        r = radius_m * np.sqrt(rng.random(N))
        a = rng.random(N) * 2 * np.pi
        th = theta_deg + rng.normal(0, self.cfg.theta_init_sigma_deg, N)
        self.x = np.column_stack([e + r * np.cos(a), n + r * np.sin(a), th % 360])
        self.w = np.full(N, 1.0 / N)

    def predict(self, fwd_m: float, right_m: float, dtheta_deg: float) -> None:
        c, rng, N = self.cfg, self.rng, len(self.w)
        th = np.radians(self.x[:, 2])
        de = fwd_m * np.sin(th) + right_m * np.cos(th)
        dn = fwd_m * np.cos(th) - right_m * np.sin(th)
        s = c.sigma_pos0_m + c.sigma_pos_rate * math.hypot(fwd_m, right_m)
        noise = rng.normal(0, s, (N, 2))
        jump = rng.random(N) < c.jump_p
        noise[jump] += rng.normal(0, c.jump_sigma_m, (int(jump.sum()), 2))
        self.x[:, 0] += de + noise[:, 0]
        self.x[:, 1] += dn + noise[:, 1]
        self.x[:, 2] = (self.x[:, 2] + dtheta_deg + rng.normal(0, c.sigma_theta_deg, N)) % 360

    def predict_en(self, de: float, dn: float, dtheta_deg: float) -> None:
        """B planı (sabit hız): harita çerçevesinde yer değiştirme; gürültü predict ile aynı."""
        c, rng, N = self.cfg, self.rng, len(self.w)
        s = c.sigma_pos0_m + c.sigma_pos_rate * math.hypot(de, dn)
        noise = rng.normal(0, s, (N, 2))
        jump = rng.random(N) < c.jump_p
        noise[jump] += rng.normal(0, c.jump_sigma_m, (int(jump.sum()), 2))
        self.x[:, 0] += de + noise[:, 0]
        self.x[:, 1] += dn + noise[:, 1]
        self.x[:, 2] = (self.x[:, 2] + dtheta_deg + rng.normal(0, c.sigma_theta_deg, N)) % 360

    def explains(self, meas: GridMeas) -> float:
        """Ölçümün bulutu ne kadar açıkladığı: ağırlıklı ortalama P / max P (0 → bulut karo tepesinden uzakta)."""
        return float(self.w @ meas.lookup(self.x[:, 0], self.x[:, 1]))

    def update(self, meas: GridMeas, theta_meas: float | None = None) -> None:
        c = self.cfg
        lik = meas.lookup(self.x[:, 0], self.x[:, 1]) ** c.lik_power + c.eps
        if theta_meas is not None and np.isfinite(theta_meas) and c.theta_meas_sigma_deg < 1e6:
            d = (self.x[:, 2] - theta_meas + 180) % 360 - 180
            lik = lik * ((1 - c.theta_outlier_p) * np.exp(-0.5 * (d / c.theta_meas_sigma_deg) ** 2) + c.theta_outlier_p)
        w = self.w * lik
        self.w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
        if self.n_eff() < c.resample_frac * len(self.w):
            self.resample()

    def n_eff(self) -> float:
        return float(1.0 / np.sum(self.w ** 2))

    def resample(self) -> None:
        N = len(self.w)
        pos = (self.rng.random() + np.arange(N)) / N
        idx = np.minimum(np.searchsorted(np.cumsum(self.w), pos), N - 1)
        self.x, self.w = self.x[idx].copy(), np.full(N, 1.0 / N)

    def estimate(self) -> dict:
        w = self.w
        e, n = float(w @ self.x[:, 0]), float(w @ self.x[:, 1])
        th = np.radians(self.x[:, 2])
        theta = math.degrees(math.atan2(w @ np.sin(th), w @ np.cos(th))) % 360
        d = np.hypot(self.x[:, 0] - e, self.x[:, 1] - n)
        return {"e": e, "n": n, "theta": theta, "spread_q90_m": wquantile(d, w, 0.9),
                "std_m": float(np.sqrt(w @ d ** 2)), "n_eff": self.n_eff()}
