"""Faz 1'in kalan bütün GPU işleri, sırayla, tek komutla (RTX 3090 makinesi).

Kullanım (repo kökünde, .venv etkin):
  python scripts/faz1_3090.py                 # hepsi, baştan sona
  python scripts/faz1_3090.py --list          # adımları ve sürelerini göster
  python scripts/faz1_3090.py --from cache_dev  # bir adımdan devam
  python scripts/faz1_3090.py --only repro    # yalnızca bir adım

Her adım kesintiye dayanıklıdır: önbellek var olan dosyaları, tekrarlama biten tohumları atlar.
Elektrik kesilir / bilgisayar kapanırsa aynı komutu yeniden çalıştırmak yeter (önce bozuk önbellek
dosyaları otomatik temizlenir). Hata olursa betik durur ve hangi adımda kaldığını yazar.
Ayrıntılı açıklama: docs/faz1_kalan_isler.md. Günlük: logs/faz1_3090.log
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
UAV = "data/UAV_VisLoc_dataset"
MD = "data/megadepth"
MD_URL = "https://github.com/Parskatt/storage/releases/download/mega1500"
MD_NPZ = ["0015_0.1_0.3.npz", "0015_0.3_0.5.npz", "0022_0.1_0.3.npz", "0022_0.3_0.5.npz",
          "0022_0.5_0.7.npz"]
SEEDS = ["0", "1", "2", "3", "4"]
LOG = ROOT / "logs" / "faz1_3090.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(*args: str, expect: str | None = None) -> None:
    """Betiği çalıştırır; çıktı hem ekrana hem günlüğe. expect: çıktıda olması gereken metin."""
    cmd = [PY, *args]
    log("$ " + " ".join(args))
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", env=env)
    out = []
    with LOG.open("a", encoding="utf-8") as f:
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
            out.append(line)
    if p.wait() != 0:
        raise RuntimeError(f"başarısız (çıkış {p.returncode}): {' '.join(args)}")
    if expect and not any(expect in x for x in out):
        raise RuntimeError(f"çıktıda '{expect}' yok: {' '.join(args)}")


# ---------------------------------------------------------------------------
def step_env():
    run("scripts/00_check_env.py")
    run("-m", "pytest", "-q")


def step_data():
    if not (ROOT / UAV).exists():
        raise RuntimeError(f"{UAV} yok: UAV-VisLoc paketini buraya kopyalayın")
    run("scripts/11_verify_data.py", UAV, "--expect-frames", "6774", "--expect-maps", "11",
        "--checksums")
    d = subprocess.run(["git", "diff", "--stat", "docs/checksums/"], check=False, cwd=ROOT,
                       capture_output=True, text=True).stdout.strip()
    if d:
        raise RuntimeError("veri dosyaları referans checksum'larla aynı değil:\n" + d)


def step_pairs():
    run("scripts/14_make_pairs.py", UAV)
    want = (ROOT / "docs/checksums/pairs_v1.sha256").read_text().split()[0]
    got = hashlib.sha256((ROOT / "data/pairs/pairs_v1.csv").read_bytes()).hexdigest()
    if got != want:
        raise RuntimeError(f"pairs_v1.csv checksum farklı: {got[:16]} ≠ {want[:16]} "
                           "(configs/pairs.yaml ya da veri farklı)")
    log("pairs_v1.csv checksum doğru")


def step_speed():
    run("scripts/23_speed.py", UAV, "--settings", "base", "mega1500")


def step_megadepth():
    md = ROOT / MD
    md.mkdir(parents=True, exist_ok=True)
    for f in MD_NPZ:
        if not (md / f).exists():
            log(f"indiriliyor: {f}")
            urllib.request.urlretrieve(f"{MD_URL}/{f}", md / (f + ".part"))
            (md / (f + ".part")).replace(md / f)
    if not (md / "Undistorted_SfM").exists():
        tar = md / "megadepth_test_1500.tar"
        if not tar.exists():
            log("indiriliyor: megadepth_test_1500.tar (birkaç GB)")
            urllib.request.urlretrieve(f"{MD_URL}/megadepth_test_1500.tar", str(tar) + ".part")
            Path(str(tar) + ".part").replace(tar)
        log("açılıyor")
        with tarfile.open(tar) as t:
            t.extractall(md, filter="data")
        (md / "megadepth_test_1500" / "Undistorted_SfM").replace(md / "Undistorted_SfM")
        (md / "megadepth_test_1500").rmdir()
        tar.unlink()
    log("MegaDepth-1500 hazır")


def step_smoke():
    run("scripts/20_repro_mega1500.py", "--data", MD, "--limit", "5", "--check-original",
        expect="birebir aynı: True")


def verify_cache():
    run("scripts/31_verify_cache.py", "cache/v1", "--delete-bad")


def step_cache_dev():
    verify_cache()
    run("scripts/30_build_cache.py", UAV, "--split", "calib")
    run("scripts/30_build_cache.py", UAV, "--split", "train")
    verify_cache()


def step_baselines():
    for sp in ("train", "calib"):
        run("scripts/40_baselines.py", "feats", "--cache", "cache/v1", "--split", sp,
            "--ransac-px", "3", "5", "8")
    run("scripts/40_baselines.py", "select", "--cache", "cache/v1")
    run("scripts/40_baselines.py", "eval", "--cache", "cache/v1", "--split", "calib")


def step_repro():
    run("scripts/20_repro_mega1500.py", "--data", MD, "--seeds", *SEEDS)
    run("scripts/20_repro_mega1500.py", "--data", MD, "--c4", "--seeds", *SEEDS)


def step_cache_rest():
    verify_cache()
    run("scripts/30_build_cache.py", UAV, "--seq")
    run("scripts/30_build_cache.py", UAV, "--split", "test")
    verify_cache()
    run("-m", "pytest", "-q", "tests/test_cache_schema.py")


STEPS = [
    ("env", step_env, "ortam + testler", "5 dk"),
    ("data", step_data, "UAV-VisLoc doğrulama + checksum", "10 dk"),
    ("pairs", step_pairs, "çift listesi (uyarlamalı karo) + checksum", "5 dk"),
    ("speed", step_speed, "hız ölçümü (süre tahminleri için)", "5 dk"),
    ("megadepth", step_megadepth, "MegaDepth-1500 test verisini indir", "10–30 dk"),
    ("smoke", step_smoke, "MegaDepth duman testi (orijinalle birebir)", "5 dk"),
    ("cache_dev", step_cache_dev, "önbellek: calib + train (Faz 2 için gereken)", "≈ 4–5 sa"),
    ("baselines", step_baselines, "taban çizgisi tablosu (CPU)", "≈ 30 dk"),
    ("repro", step_repro, "MegaDepth-1500 + C4, 5'er tohum", "≈ 10–15 sa"),
    ("cache_rest", step_cache_rest, "önbellek: ardışık kareler + test", "≈ 3 sa"),
]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=[s[0] for s in STEPS])
    ap.add_argument("--only", choices=[s[0] for s in STEPS])
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for i, (name, _, desc, dur) in enumerate(STEPS, 1):
            print(f"{i:2d}. {name:<11} {desc} ({dur})")
        return 0
    names = [s[0] for s in STEPS]
    todo = STEPS[names.index(a.start):] if a.start else STEPS
    if a.only:
        todo = [s for s in STEPS if s[0] == a.only]
    t0 = time.time()
    for name, fn, desc, _ in todo:
        log(f"=== {name}: {desc} ===")
        ts = time.time()
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — kullanıcıya nerede kaldığını söyle
            log(f"!!! {name} adımında durdu: {e}")
            log(f"Sorunu giderip devam: python scripts/faz1_3090.py --from {name}")
            return 1
        log(f"=== {name} bitti ({(time.time() - ts) / 60:.0f} dk) ===")
    log(f"HEPSİ BİTTİ ({(time.time() - t0) / 3600:.1f} sa). Sonraki adım: docs/faz1_kalan_isler.md §4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
