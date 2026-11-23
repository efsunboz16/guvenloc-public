import math

from guvenloc.geo import GeoRef, angular_error_deg

REF = GeoRef(lat_top=27.0, lon_left=120.0, lat_bottom=26.9, lon_right=120.1,
             width_px=30000, height_px=33000)


def test_roundtrip_under_1cm():
    mx, my = REF.meters_per_px()
    for x, y in [(0, 0), (15000.5, 16500.25), (29999, 32999), (123.4, 31000.9)]:
        lat, lon = REF.px_to_latlon(x, y)
        x2, y2 = REF.latlon_to_px(lat, lon)
        assert abs(x2 - x) * mx < 0.01
        assert abs(y2 - y) * my < 0.01


def test_meters_per_px_plausible():
    mx, my = REF.meters_per_px()
    assert 0.2 < mx < 0.5 and 0.2 < my < 0.5


def test_angular_error_wraps():
    assert angular_error_deg(359, 1) == 2
    assert angular_error_deg(10, 350) == 20
    assert angular_error_deg(180, 0) == 180
    assert math.isclose(angular_error_deg(90, 90), 0)


def test_north_up_rotation_moves_heading_to_north():
    import numpy as np

    from guvenloc.geo import north_up_rotation_cw_deg

    for phi in (-91.0, -39.3, 0.0, 45.0, 164.0):
        rot = math.radians(north_up_rotation_cw_deg(phi))
        # Görüntü 'yukarısı' (0,-1) haritada pusula yönü phi'ye bakar: (sin φ, -cos φ) y-aşağı
        up_in_map = np.array([math.sin(math.radians(phi)), -math.cos(math.radians(phi))])
        # Haritadaki bu yönü görüntü çerçevesine geri al ve kare saat yönünde 'rot' döndür:
        # sonuç görüntü yukarısı tekrar kuzey olmalı → harita çerçevesi = kuzey-yukarı kare.
        R_cw = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
        np.testing.assert_allclose(R_cw @ np.array([0.0, -1.0]), up_in_map, atol=1e-9)
