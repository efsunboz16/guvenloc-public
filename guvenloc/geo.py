"""Coğrafi dönüşümler: uydu pikseli <-> enlem/boylam <-> metre.

Uydu haritasının sol-üst ve sağ-alt köşe koordinatları (satellite_coordinates_range.csv)
bilindiğinde, küçük alanlar için doğrusal (eşdikdörtgen) eşleme kullanılır. Bölge
birkaç yüz km² olduğundan hata ihmal edilebilir; gerçek georeferans (rasterio) ile
`scripts/12_overlay_check.py` aşamasında karşılaştırılacaktır.

DOĞRULANMADI: UAV-VisLoc'taki yön (Phi) sütununun işareti, referansı ve birimi.
Bindirme kontrolünden sonra buraya yazılacaktır (MIMARI.md 1B).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

EARTH_R = 6_371_008.8  # m


@dataclass(frozen=True)
class GeoRef:
    """Uydu haritasının doğrusal georeferansı."""
    lat_top: float
    lon_left: float
    lat_bottom: float
    lon_right: float
    width_px: int
    height_px: int

    def latlon_to_px(self, lat: float, lon: float) -> tuple[float, float]:
        x = (lon - self.lon_left) / (self.lon_right - self.lon_left) * self.width_px
        y = (self.lat_top - lat) / (self.lat_top - self.lat_bottom) * self.height_px
        return x, y

    def px_to_latlon(self, x: float, y: float) -> tuple[float, float]:
        lon = self.lon_left + x / self.width_px * (self.lon_right - self.lon_left)
        lat = self.lat_top - y / self.height_px * (self.lat_top - self.lat_bottom)
        return lat, lon

    def meters_per_px(self) -> tuple[float, float]:
        """(x yönünde, y yönünde) piksel başına metre."""
        lat_mid = math.radians((self.lat_top + self.lat_bottom) / 2)
        width_m = math.radians(self.lon_right - self.lon_left) * EARTH_R * math.cos(lat_mid)
        height_m = math.radians(self.lat_top - self.lat_bottom) * EARTH_R
        return width_m / self.width_px, height_m / self.height_px


def angular_error_deg(pred: float, gt: float) -> float:
    """Dairesel yön hatası: min(|θ̂ − φ|, 360° − |θ̂ − φ|)."""
    d = abs(pred - gt) % 360.0
    return min(d, 360.0 - d)
