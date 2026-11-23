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
