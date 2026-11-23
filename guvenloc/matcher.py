"""RoMa v2 sarmalayıcı (MIMARI.md 1E). RoMa v2 DONUKTUR: yalnızca çıkarım.

Hassasiyet birimi (Faz 0.4, scripts/02_precision_units.py, results/faz0/):
  RoMa v2'nin `precision_AB` haritası, refiner'ın SON aşamasının ızgara PİKSELİNDE
  tanımlıdır (hr varsa hr, yoksa lr: base → 640, mega1500 → 1024). Normalize [-1,1]
  koordinatı DEĞİL (o varsayım σ'yı ~1000 kat büyük veriyordu). Görüntü pikseline:
      P_img = D · P_grid · D,   D = diag(W_grid / W_img, H_grid / H_img)
  (= RoMaV2.prec_map_coordinates). Kolay sentetik çiftte σ ≈ 0,2–0,3 px, gerçek hata
  ≈ 0,06–0,08 px: aynı mertebe, model temkinli. Gerçek drone↔uydu verisi için ölçek
  katsayısı `sigma_scale` (Σ ← sigma_scale² · Σ) eğitim bölgelerinde NLL ile seçilir.

Koordinat kuralı: OpenCV — piksel merkezi tamsayı (RoMa'nın to_pixel çıktısı − 0,5).

Örnekleme: `balanced_sample` RoMaV2.sample() ile AYNI algoritmadır (güvenle orantılı 4N
aday → KDE yoğunluğuyla dengeleme → N), yalnızca KDE parça parça hesaplanır. Orijinal
40k×40k mesafe matrisini tek seferde kuruyor (≈3,2 GB fp16) ve 8 GB kartta eşleştirmenin
4 katı sürüyordu (1,7 s → ~0,1 s).
romav2 2.0.1 `sample()` hatası: tek yönlü ayarlarda hassasiyeti (N,2) döndürür. Burada
hassasiyetler `sample()`'dan değil, yoğun haritalardan eşleşme konumunda okunur:
  P_B (v'nin belirsizliği) = precision_AB, A'daki konumda
  P_A (u'nun belirsizliği) = precision_BA, B'deki konumda (yalnızca çift yönlü ayarlarda)
Tek yönlü ayarlarda eşleşmenin A ucu ızgara noktasıdır → Σ_A = 0.

`rot_deg`: drone karesine SAAT YÖNÜNDE uygulanan dönme (geo.north_up_rotation_cw_deg ile
aynı kural). 90'ın katları birebir (np.rot90); diğer açılar genişletilmiş tuvalle
(warpAffine). `u` ve `Sigma_A` her zaman ORİJİNAL kare koordinatına geri çevrilir.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROMAV2_VERSION = "2.0.1"
NPZ_FIELDS = ("u", "v", "c", "Sigma_A", "Sigma_B")
# Saklanan Σ'nın özdeğer aralığı (px²): simetrikleştirme + pozitif tanımlılık güvencesi.
# Σ float32 saklanır: fp16 hem taşıyor (σ > 256 px) hem de anizotrop Σ'da (göreli hassasiyet
# ≈ 5e-4) küçük özdeğeri negatife itiyordu. σ ≤ 1000 px; Σ_A = 0 tek yönlü ayarlarda 0 kalır.
SIGMA_EIG_RANGE_PX2 = (1e-4, 1.0e6)


def clamp_cov(S: np.ndarray, lo: float = SIGMA_EIG_RANGE_PX2[0],
              hi: float = SIGMA_EIG_RANGE_PX2[1]) -> np.ndarray:
    """Simetrikleştir ve özdeğerleri [lo, hi] aralığına kıstır; sıfır matrisler sıfır kalır."""
    S = 0.5 * (S + np.swapaxes(S, -1, -2)).astype(np.float64)
    zero = np.all(S == 0, axis=(-1, -2))
    w, V = np.linalg.eigh(S)
    w = np.clip(w, lo, hi)
    out = (V * w[..., None, :]) @ np.swapaxes(V, -1, -2)
    out[zero] = 0.0
    return out


@dataclass
class MatchSet:
    u: np.ndarray            # (N,2) float32, drone karesi pikseli (orijinal, döndürülmemiş)
    v: np.ndarray            # (N,2) float32, uydu karosu pikseli
    c: np.ndarray            # (N,)  float32, RoMa v2 örtüşme güveni (sigmoid)
    Sigma_A: np.ndarray      # (N,2,2) float32, u'nun kovaryansı, drone pikseli²
    Sigma_B: np.ndarray      # (N,2,2) float32, v'nin kovaryansı, karo pikseli²
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.u)

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        meta = {**self.meta, "sigma_eig_range_px2": list(SIGMA_EIG_RANGE_PX2)}
        np.savez_compressed(
            tmp, u=self.u.astype(np.float32), v=self.v.astype(np.float32),
            c=self.c.astype(np.float16), Sigma_A=clamp_cov(self.Sigma_A).astype(np.float32),
            Sigma_B=clamp_cov(self.Sigma_B).astype(np.float32), meta=json.dumps(meta),
        )
        tmp.replace(path)  # yarım yazılmış dosya bırakma (önbellek yeniden başlatılabilir)

    @classmethod
    def load_npz(cls, path: str | Path) -> MatchSet:
        with np.load(path) as z:
            return cls(z["u"], z["v"], z["c"].astype(np.float32),
                       z["Sigma_A"].astype(np.float32), z["Sigma_B"].astype(np.float32),
                       json.loads(str(z["meta"])))


# ---------------------------------------------------------------------------
# Dönme geometrisi
# ---------------------------------------------------------------------------
def rotation_affine(w: int, h: int, rot_deg: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Orijinal (w×h) → döndürülmüş görüntü pikseli 2×3 afin matrisi ve yeni boyut (w', h').

    Saat yönünde rot_deg. 90'ın katlarında np.rot90 ile birebir aynı piksel yerleşimi.
    """
    r = rot_deg % 360.0
    if math.isclose(r % 90.0, 0.0, abs_tol=1e-9) or math.isclose(r % 90.0, 90.0, abs_tol=1e-9):
        k = round(r / 90.0) % 4
        if k == 0:
            return np.array([[1.0, 0, 0], [0, 1.0, 0]]), (w, h)
        if k == 1:   # saat yönü 90: (x, y) → (h-1-y, x)
            return np.array([[0, -1.0, h - 1], [1.0, 0, 0]]), (h, w)
        if k == 2:   # (x, y) → (w-1-x, h-1-y)
            return np.array([[-1.0, 0, w - 1], [0, -1.0, h - 1]]), (w, h)
        return np.array([[0, 1.0, 0], [-1.0, 0, w - 1]]), (h, w)  # 270: (x, y) → (y, w-1-x)
    t = math.radians(r)
    cs, sn = math.cos(t), math.sin(t)
    W2 = math.ceil(abs(w * cs) + abs(h * sn))
    H2 = math.ceil(abs(w * sn) + abs(h * cs))
    # y aşağı koordinatta saat yönü dönme: [[c, -s], [s, c]]
    R = np.array([[cs, -sn], [sn, cs]])
    c0 = np.array([(w - 1) / 2, (h - 1) / 2])
    c1 = np.array([(W2 - 1) / 2, (H2 - 1) / 2])
    return np.hstack([R, (c1 - R @ c0)[:, None]]), (W2, H2)


def rotate_image(img: np.ndarray, rot_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """(döndürülmüş görüntü, orijinal→döndürülmüş afin)."""
    h, w = img.shape[:2]
    M, (W2, H2) = rotation_affine(w, h, rot_deg)
    r = rot_deg % 360.0
    if math.isclose(r % 90.0, 0.0, abs_tol=1e-9) or math.isclose(r % 90.0, 90.0, abs_tol=1e-9):
        k = round(r / 90.0) % 4
        return np.ascontiguousarray(np.rot90(img, -k)), M   # np.rot90 saat yönü tersi
    out = cv2.warpAffine(img, M, (W2, H2), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, M


def unrotate_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    Minv = cv2.invertAffineTransform(M)
    return pts @ Minv[:, :2].T + Minv[:, 2]


def unrotate_cov(S: np.ndarray, M: np.ndarray) -> np.ndarray:
    L = cv2.invertAffineTransform(M)[:, :2]
    return L @ S @ L.T


def precision_grid_to_image(P: np.ndarray, grid_wh, img_wh) -> np.ndarray:
    """Izgara pikselindeki hassasiyeti görüntü pikseline taşır (Faz 0.4)."""
    D = np.diag([grid_wh[0] / img_wh[0], grid_wh[1] / img_wh[1]])
    return D @ P @ D


def safe_inv(P: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    P = 0.5 * (P + np.swapaxes(P, -1, -2))
    return np.linalg.inv(P + eps * np.eye(2))


# ---------------------------------------------------------------------------
# Dengeli örnekleme (RoMaV2.sample ile aynı algoritma, parça parça KDE)
# ---------------------------------------------------------------------------
def kde_chunked(x, std: float = 0.1, chunk: int = 4096):
    """romav2.kde ile aynı (fp16, Gauss çekirdeği), satır parçaları hâlinde."""
    import torch

    xh = x.half()
    out = torch.empty(len(x), device=x.device, dtype=torch.float16)  # orijinal gibi fp16
    for i in range(0, len(x), chunk):
        d2 = torch.cdist(xh[i:i + chunk], xh) ** 2
        out[i:i + chunk] = (-d2 / (2 * std**2)).exp().sum(dim=-1)
    return out


def balanced_sample(preds: dict, num_corresp: int, bidirectional: bool):
    """RoMaV2.sample()'ın eşleşme/güven kısmı: (matches (N,4) normalize, confidence (N,))."""
    import torch
    from romav2.geometry import get_normalized_grid

    warp = preds["warp_AB"][0]
    H_A, W_A, _ = warp.shape
    grid = get_normalized_grid(1, H_A, W_A, overload_device=warp.device)[0]
    matches = torch.cat((grid, warp), dim=-1).reshape(-1, 4)
    confidence = preds["overlap_AB"][0].reshape(-1)
    if bidirectional:
        matches_BA = torch.cat((preds["warp_BA"][0], grid), dim=-1).reshape(-1, 4)
        matches = torch.cat((matches, matches_BA), dim=0)
        confidence = torch.cat((confidence, preds["overlap_BA"][0].reshape(-1)), dim=0)
    expansion_factor = 4
    confidence = confidence * matches.abs().amax(dim=-1).le(1 - 1 / H_A).float()
    n_cand = min(expansion_factor * num_corresp, int((confidence > 0).sum()))
    if n_cand == 0:
        return matches[:0], confidence[:0]
    inds = torch.multinomial(confidence, n_cand, replacement=False)
    sm, sc = matches[inds], confidence[inds]
    density = kde_chunked(sm)
    p = 1 / (density + 1)
    p[density < 10] = 1e-7  # RoMa: en az ~10 iyi komşu
    bal = torch.multinomial(p, num_samples=min(num_corresp, len(sc)), replacement=False)
    return sm[bal], sc[bal]


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------
class Matcher:
    def __init__(self, setting: str = "mega1500", sigma_scale: float = 1.0, seed: int = 0):
        import torch
        from romav2 import RoMaV2

        torch.set_float32_matmul_precision("highest")  # RoMaV2.forward şart koşuyor
        self.torch = torch
        self.setting = setting
        self.sigma_scale = float(sigma_scale)
        self.seed = seed
        self.model = RoMaV2()
        self.model.apply_setting(setting)
        m = self.model
        self.grid_wh = (m.W_hr or m.W_lr, m.H_hr or m.H_lr)

    def _grid_sample(self, dense, coords_n):
        from romav2.geometry import bhwc_grid_sample

        h, w = dense.shape[1:3]
        out = bhwc_grid_sample(dense.reshape(1, h, w, -1), coords_n[None, None],
                               mode="bilinear", align_corners=False)
        return out.reshape(coords_n.shape[0], 2, 2)

    def match(self, img_drone: np.ndarray, img_sat: np.ndarray, rot_deg: float = 0.0,
              n: int = 10_000) -> MatchSet:
        """img_*: (H,W,3) uint8 RGB. Dönüş: orijinal drone koordinatında MatchSet."""
        torch = self.torch
        from romav2.geometry import to_pixel

        h, w = img_drone.shape[:2]
        Hs, Ws = img_sat.shape[:2]
        A, M = rotate_image(np.ascontiguousarray(img_drone), rot_deg)
        Ha, Wa = A.shape[:2]
        torch.manual_seed(self.seed)  # sample() rastgele; aynı girdi → aynı eşleşmeler
        preds = self.model.match(A, np.ascontiguousarray(img_sat))
        matches, overlaps = balanced_sample(preds, n, self.model.bidirectional)
        PB = self._grid_sample(preds["precision_AB"], matches[:, :2])
        PA = (self._grid_sample(preds["precision_BA"], matches[:, 2:])
              if preds.get("precision_BA") is not None else None)

        ua = to_pixel(matches[:, :2], H=Ha, W=Wa).float().cpu().numpy() - 0.5
        v = to_pixel(matches[:, 2:], H=Hs, W=Ws).float().cpu().numpy() - 0.5
        c = overlaps.float().cpu().numpy().reshape(-1)
        k2 = self.sigma_scale ** 2
        SB = k2 * safe_inv(precision_grid_to_image(PB.double().cpu().numpy(),
                                                   self.grid_wh, (Ws, Hs)))
        if PA is not None:
            SA_rot = k2 * safe_inv(precision_grid_to_image(PA.double().cpu().numpy(),
                                                           self.grid_wh, (Wa, Ha)))
            SA = unrotate_cov(SA_rot, M)
        else:
            SA = np.zeros_like(SB)
        u = unrotate_points(ua, M)
        # genişletilmiş tuvalde orijinal karenin dışına düşen eşleşmeleri at
        keep = (u[:, 0] >= -0.5) & (u[:, 0] <= w - 0.5) & (u[:, 1] >= -0.5) & (u[:, 1] <= h - 0.5)
        meta = {"setting": self.setting, "romav2": ROMAV2_VERSION, "rot_deg": float(rot_deg),
                "drone_wh": [w, h], "sat_wh": [Ws, Hs], "grid_wh": list(self.grid_wh),
                "sigma_scale": self.sigma_scale, "bidirectional": PA is not None,
                "coords": "opencv (piksel merkezi tamsayı)", "cov_units": "görüntü pikseli²"}
        return MatchSet(u[keep].astype(np.float32), v[keep].astype(np.float32),
                        c[keep].astype(np.float32), SA[keep].astype(np.float32),
                        SB[keep].astype(np.float32), meta)
