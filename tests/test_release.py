"""Faz 6D – sürüm paketi denetimi: görüntü/dizi ve yerel yol içeren dosyalar reddedilir (F6-6)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("rel", Path(__file__).parents[1] / "scripts" / "95_release.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def test_forbidden_and_paths(tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"x")
    assert any("yasak" in e for e in R.check_file(img, "meta/a.jpg"))
    arr = tmp_path / "f.csv"
    arr.write_text("ok\n")
    assert any("yasak" in e for e in R.check_file(arr, "meta/f.npy"))
    assert R.check_file(arr, "meta/f.csv") == []
    bad = tmp_path / "p.json"
    bad.write_text('{"p": "/home/kisi/veri"}')
    assert any("yerel yol" in e for e in R.check_file(bad, "params/p.json"))
    assert R.check_file(tmp_path / "yok.csv", "meta/yok.csv") == [f"eksik: {tmp_path / 'yok.csv'}"]


def test_package_list_has_no_pixels():
    assert all(Path(d).suffix not in R.FORBIDDEN_EXT for _, d in R.FILES)
    assert R.UNPACK == {d: s for s, d in R.FILES}
