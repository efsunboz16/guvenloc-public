"""Faz 6 – test bölgeleri için dondurma ve tek seferlik kilit (MIMARI 6A: "tek sefer, dondurulmuş config ile").

  freeze(paths)  → configs/final_manifest.json: her dosyanın sha256'sı + git commit'i (test koşmadan ÖNCE commit edilir)
  check(name)    → manifestteki her dosya aynı mı (değilse durur); results/faz6/final/locks/<name>.done varsa durur
  done(name)     → kilidi yazar; aynı değerlendirme ikinci kez koşulamaz

Kilit, "testi görüp ayarı değiştir, yeniden koş" yolunu kapatır. Bir hata yüzünden yeniden koşmak gerekirse kilit
elle silinir ve bu, docs/faz6_test.md'ye gerekçesiyle karar olarak yazılır.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

MANIFEST = Path("configs/final_manifest.json")
LOCKS = Path("results/faz6/final/locks")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def freeze(paths: list[str]) -> dict:
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        sys.exit(f"dondurulamaz, eksik: {missing}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    man = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": commit,
           "files": {p: sha256(Path(p)) for p in sorted(paths)}}
    MANIFEST.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return man


def check(name: str) -> dict:
    if not MANIFEST.exists():
        sys.exit("test bölgeleri kapalı: önce dondurma (scripts/81_freeze.py)")
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    bad = [p for p, h in man["files"].items() if not Path(p).exists() or sha256(Path(p)) != h]
    if bad:
        sys.exit(f"dondurulmuş dosyalar değişmiş, test koşulamaz: {bad}")
    if (LOCKS / f"{name}.done").exists():
        sys.exit(f"{name} test değerlendirmesi zaten koşuldu ({LOCKS / f'{name}.done'}); tek sefer")
    return man


def done(name: str) -> None:
    LOCKS.mkdir(parents=True, exist_ok=True)
    (LOCKS / f"{name}.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
