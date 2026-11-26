"""Faz 4G – uçtan uca konumlandırma: `localize(frame, tile) -> Decision` (MIMARI.md 4G, satır 6 = h3_v1).

Akış (tek kare × tek karo, 1 RoMa çağrısı):
  1. Dönme: M1-C'nin 4 dönme olasılığı (rot_probs) → en olası r (satır 4a). Ya da r doğrudan verilir.
  2. RoMa v2 (drone r° saat yönünde) → MatchSet (Σ ham, meta.sigma_scale = 1; önbellek cache/v1 ile aynı).
  3. RANSAC özellikleri (baselines.features, 5 px) → n_inliers;  oylama hacmi (m2_volume) → tepe (x̂, ŷ, θ̂),
     yayılım spread_m, destek, ρ, entropi …
  4. S = h2_v3 lojistik skoru (61_m2_score.feat_matrix ile aynı öznitelikler).
  5. Conformal (m3_conformal): kabul ⇔ S > τ;  σ̂ = max(spread_m (NaN → 50), 2 m);  yarıçap = q̂ · σ̂.

Parametreler (S ağırlıkları, τ, q̂) `scripts/67_m3_pipeline.py --export` ile dosyaya yazılır:
S eğitim bölgelerinde (h2_v3), τ ve q̂ kalibrasyon bölgelerinde (configs/h3.yaml calibration_regions).
M1-C ağı bu pakette değil (eğitim betiklerine bağlı); `rot_probs` dışarıdan verilir (67'deki M1Live).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from guvenloc import baselines as B
from guvenloc import m2_volume as V
from guvenloc.geo import EARTH_R
from guvenloc.m3_conformal import Calibration
from guvenloc.matcher import MatchSet

ROTS = (0, 90, 180, 270)
SPREAD_NAN_M = 50.0


@dataclass
class Decision:
    xy: tuple[float, float]           # karo pikseli (OpenCV: piksel merkezi tamsayı)
    latlon: tuple[float, float] | None
    theta: float                      # görüntü üst kenarının pusula yönü (°, saat yönü), hacim tepesinden
    sigma_hat: float                  # m
    score: float                      # S (logit)
    accept: bool
    radius_m: float                   # q̂ · σ̂ (kabul koşullu %(1 − β) güven yarıçapı); ret ise yine hesaplanır
    rot: int                          # RoMa'ya verilen dönme (°)
    n_inliers: int

    def to_dict(self) -> dict:
        return asdict(self)


def feature_row(ms: MatchSet, scale_s: float, tile_px: int, mpp: float, vcfg: V.VolumeCfg,
                ransac_px: float = 5.0) -> dict:
    """Tek (çift, dönme) için 60_m2_features + 40_baselines satırının aynısı."""
    r = B.features(ms, ransac_px=ransac_px)
    vm = V.metrics(V.build_volume(ms.u, ms.v, ms.c, ms.Sigma_B, ms.Sigma_A, float(scale_s), ms.meta["drone_wh"],
                                  int(tile_px), float(mpp), vcfg))
    return vm | {"n_inliers": int(r["n_inliers"]), "cx": r["cx"], "cy": r["cy"], "valid": r["valid"]}


def feat_vector(f: dict, names, entropy_fill: float) -> np.ndarray:
    """61_m2_score.feat_matrix'in tek satırlık karşılığı; entropy NaN → eğitim kümesinin en büyüğü (entropy_fill)."""
    ent = f["entropy_bits"]
    cols = {"log1p_support": math.log1p(max(f["support"], 0.0)), "log1p_sum_c": math.log1p(f["sum_c"]),
            "support_frac": f["support_frac"], "rho": f["rho"],
            "spread_m": SPREAD_NAN_M if not np.isfinite(f["spread_m"]) else f["spread_m"],
            "entropy_bits": ent if np.isfinite(ent) else entropy_fill, "frac_dropped": f["frac_dropped"],
            "log1p_inliers": math.log1p(max(f["n_inliers"], 0))}
    return np.array([float(cols[n]) for n in names])


def tile_px_to_latlon(x: float, y: float, tile_lat: float, tile_lon: float, tile_size_m: float,
                      tile_px: int) -> tuple[float, float]:
    """Karo (crop_square_m: merkez (lat, lon), kuzey-yukarı, kenar = tile_size_m) pikselinden enlem/boylam.
    Piksel kenarı = 0 kuralı → merkez tile_px / 2; OpenCV merkezli koordinat için +0,5."""
    mpp = tile_size_m / tile_px
    de, dn = (x + 0.5 - tile_px / 2) * mpp, -(y + 0.5 - tile_px / 2) * mpp
    lat = tile_lat + math.degrees(dn / EARTH_R)
    return lat, tile_lon + math.degrees(de / (EARTH_R * math.cos(math.radians(tile_lat))))


class Localizer:
    def __init__(self, params: dict, matcher=None, n_matches: int = 10_000):
        self.p = params
        self.cal = Calibration(tau=float(params["tau"]), q=np.asarray(params["q"], float),
                               edges=np.asarray(params.get("edges", []), float))
        self.score_model = {k: np.asarray(params["score_model"][k], float) for k in ("w", "mu", "sd")}
        self.features = list(params["score_model"]["features"])
        self.vcfg = V.VolumeCfg(**params["volume_cfg"])
        self.matcher = matcher
        self.n_matches = n_matches

    @classmethod
    def load(cls, path: str | Path, matcher=None, n_matches: int = 10_000) -> Localizer:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), matcher, n_matches)

    def score(self, f: dict) -> float:
        m = self.score_model
        z = (feat_vector(f, self.features, float(self.p["entropy_fill"])) - m["mu"]) / m["sd"]
        return float(m["w"][0] + z @ m["w"][1:])

    def decide(self, f: dict, rot: int, mpp: float, geo: dict | None = None) -> Decision:
        """Öznitelik satırından karar (RoMa gerektirmez; önbellekle tutarlılık testi buradan)."""
        s = self.score(f)
        sig = max(SPREAD_NAN_M if not np.isfinite(f["spread_m"]) else float(f["spread_m"]),
                  float(self.p["sigma_floor_m"]))
        x, y = float(f["x"]), float(f["y"])
        ll = None
        if geo is not None and np.isfinite(x):
            ll = tile_px_to_latlon(x, y, geo["tile_lat"], geo["tile_lon"], geo["tile_size_m"], geo["tile_px"])
        return Decision(xy=(x, y), latlon=ll, theta=float(f["theta"]), sigma_hat=sig, score=s,
                        accept=bool(self.cal.accept(np.array([s]))[0]),
                        radius_m=float(self.cal.radius(np.array([sig]))[0]), rot=int(rot),
                        n_inliers=int(f["n_inliers"]))

    def localize(self, frame: np.ndarray, tile: np.ndarray, scale_s: float, tile_size_m: float,
                 rot: int | None = None, rot_probs=None, geo: dict | None = None) -> Decision:
        """frame: drone karesi (H, W, 3) uint8 RGB, tam çözünürlük; tile: kuzey-yukarı uydu karosu (RGB, kare).
        scale_s: drone pikseli → karo pikseli; tile_size_m: karo kenarı (m). geo: {tile_lat, tile_lon, ...}."""
        if rot is None:
            rot = ROTS[int(np.argmax(rot_probs))] if rot_probs is not None else 0
        tile_px = tile.shape[0]
        ms = self.matcher.match(frame, tile, rot, self.n_matches)
        mpp = tile_size_m / tile_px
        f = feature_row(ms, scale_s, tile_px, mpp, self.vcfg, float(self.p.get("ransac_px", 5.0)))
        g = None if geo is None else geo | {"tile_size_m": tile_size_m, "tile_px": tile_px}
        return self.decide(f, rot, mpp, g)
