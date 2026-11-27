"""Faz 6 – dondurma ve tek seferlik kilit: değişen dosya ve ikinci koşu reddedilir."""
from __future__ import annotations

import pytest

from guvenloc import final_guard as G


def test_freeze_check_done(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(G, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(G, "LOCKS", tmp_path / "locks")
    a = tmp_path / "a.txt"
    a.write_text("bir")
    with pytest.raises(SystemExit, match="önce dondurma"):
        G.check("x")
    G.freeze(["a.txt"])
    G.check("x")                                  # değişmemiş, kilit yok → geçer
    G.done("x")
    with pytest.raises(SystemExit, match="zaten koşuldu"):
        G.check("x")
    G.check("y")                                  # başka değerlendirmenin kilidi ayrı
    a.write_text("iki")
    with pytest.raises(SystemExit, match="değişmiş"):
        G.check("y")


def test_freeze_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(G, "MANIFEST", tmp_path / "manifest.json")
    with pytest.raises(SystemExit, match="eksik"):
        G.freeze(["yok.txt"])
