"""Coğrafi dönüşümler: uydu pikseli <-> enlem/boylam <-> metre.

Georeferans uydu raster'ının kendisinden okunur (io_visloc.load_region); EPSG:4326
ızgarasında küçük alanlar için doğrusal (eşdikdörtgen) eşleme kullanılır.

YÖN TANIMI — Faz 1B'de RoMa v2 ile doğrulandı (scripts/13_roma_heading_scale.py,
10 bölge × 12 kare, docs/faz1_veri_notlari.md):
  Phi1 = drone görüntüsünün ÜST kenarının pusula yönü: derece, kuzey = 0, doğu = 90
  (saat yönünde pozitif), aralık [-180, 180). İşaret ters değil, 180° kayma yok.
  RoMa ölçümüyle fark tipik olarak −2…−4° (yayılım ≈ 3°); bazı uçuş bacaklarında
  (01, 08) 25°'ye kadar çıkıyor. Phi2 daha gürültülü.
  Kuzey-yukarı kanonik kare: görüntü SAAT YÖNÜNDE Phi1 kadar döndürülür
  (`north_up_rotation_cw_deg`). cv2.getRotationMatrix2D saat yönü tersini pozitif
  aldığından ona `-north_up_rotation_cw_deg(phi1)` verilir.
ÖLÇEK: `height` 01/03/04/08/09'da yere göre (ayak izi genişliği ≈ 1,0 × height, yatay
  görüş açısı ≈ 53°), 05/06/10/11'de deniz seviyesine göre → ölçek s height'tan değil
  bölge başına RoMa ölçümünden bağlanır.
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


def north_up_rotation_cw_deg(phi1: float) -> float:
    """Drone karesini kuzey-yukarı yapmak için gereken SAAT YÖNÜ dönme (derece, [0, 360)).

    Phi1, görüntü üst kenarının pusula yönüdür; görüntüyü saat yönünde Phi1 döndürmek
    üst kenarı kuzeye getirir (MIMARI.md 2A'daki φ etiketi budur).
    """
    return phi1 % 360.0
