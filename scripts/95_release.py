"""Faz 6D — açık kaynak sürüm paketi (v1.0): ağırlıklar + meta veri, görüntü YOK (F6-6).

  python scripts/95_release.py            # → dist/guvenloc-v1.0-paket/ + .tar.gz + SHA256SUMS
  python scripts/95_release.py --check    # yalnız içerik denetimi, yazmaz

Pakete giren her dosya aşağıdaki listede adıyla durur; veri kümesinin piksel içeren hiçbir türevi (drone/uydu
görüntüsü, karo, kırpıntı, data/m1/*.npy, cache/) pakete girmez. Ağırlık dosyaları yalnız tensör ve küçük bir cfg
sözlüğü içerir (yol, kullanıcı adı yok); bu da denetlenir. Paket kodu içermez (kod git deposunda, MIT).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "v1.0"
ENS = ["D2_olcek_rot90", "D2_olcek_rot90_s1", "D2_olcek_rot90_s2"]

# (kaynak, paketteki yol)
FILES: list[tuple[str, str]] = [
    *[(f"cache/models/m1c_{m}.pt", f"weights/m1c_{m}.pt") for m in ENS],
    *[(f"results/faz2/m1/q_sel_{m}.json", f"weights/q_sel_{m}.json") for m in [*ENS, "+".join(ENS) + "@tta"]],
    ("configs/split.yaml", "meta/split.yaml"),
    ("data/pairs/pairs_v1.csv", "meta/pairs_v1.csv"),
    ("data/pairs/decoy_v1.csv", "meta/decoy_v1.csv"),
    ("data/m1/labels_v2.csv", "meta/m1_labels_v2.csv"),
    ("results/faz2/label_correction.json", "meta/label_correction.json"),
    ("results/faz4/m3/m3_params_h3_v1.json", "params/m3_params_h3_v1.json"),
    ("configs/final_manifest.json", "params/final_manifest.json"),
]
FORBIDDEN_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".npy", ".npz", ".h5", ".webp", ".bmp"}
ALLOWED_PT_KEYS = {"state_dict", "lr", "cfg", "sat_pool"}

README = """# GüvenLoc {v} — sürüm paketi

Kod: https://github.com/efsunboz16/guvenloc (MIT, `LICENSE`). Bu paket kodu değil, kodun yeniden üretim için
gerektirdiği ve veri kümesinden yeniden üretilmesi uzun süren dosyaları içerir. **Görüntü içermez** (F6-6).

| Klasör | İçerik | Depodaki yeri |
|---|---|---|
| `weights/` | M1-C birleşimi (3 tohum, C8-eşdeğer CNN, ≈ 52 MB her biri) + seçilmiş q eşikleri | `cache/models/`, `results/faz2/m1/` |
| `meta/` | bölge ayrımı, çift listesi (dosya adı + geometri), tuzak çiftleri, M1 etiketleri, etiket düzeltmesi | `configs/`, `data/pairs/`, `data/m1/`, `results/faz2/` |
| `params/` | M3 conformal parametreleri (S, τ, q̂), test öncesi dondurma manifesti | `results/faz4/m3/`, `configs/` |

Kurulum: dosyaları "Depodaki yeri" sütunundaki yerlere kopyalayın (`python scripts/95_release.py --unpack PAKET`
bunu yapar), sonra depo README'sindeki komutlar. Bütünlük: `sha256sum -c SHA256SUMS`.

**Veri:** UAV-VisLoc (Xu vd., 2024, https://github.com/IntelliSensing/UAV-VisLoc) ayrıca indirilir; `meta/`
dosyaları o veri kümesinin dosya adlarına ve GPS kayıtlarına dayanır, kullanımda veri kümesine atıf yapılmalıdır.
Veri kümesinin kendi lisans koşulları belirtilmemiştir; ağırlıklar ve meta veri yalnız araştırma amaçlı paylaşılır.
"""

UNPACK = {dst: src for src, dst in FILES}               # paketteki yol → depodaki yol


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_file(src: Path, dst: str) -> list[str]:
    """Pakete girecek bir dosyanın sorunlarını döndürür (boş = temiz)."""
    errs = []
    if not src.exists():
        return [f"eksik: {src}"]
    if Path(dst).suffix.lower() in FORBIDDEN_EXT or src.suffix.lower() in FORBIDDEN_EXT:
        errs.append(f"görüntü/dizi uzantısı yasak: {dst}")
    if src.suffix == ".pt":
        import torch
        d = torch.load(src, map_location="cpu", weights_only=False)
        extra = set(d) - ALLOWED_PT_KEYS
        if extra:
            errs.append(f"{dst}: beklenmeyen anahtarlar {sorted(extra)}")
        blob = json.dumps({k: v for k, v in d.items() if k != "state_dict"}, default=str)
        if "/home/" in blob or "\\Users\\" in blob:
            errs.append(f"{dst}: yerel yol içeriyor")
    elif src.suffix in {".csv", ".json", ".yaml"}:
        txt = src.read_text(encoding="utf-8")
        if "/home/" in txt:
            errs.append(f"{dst}: yerel yol içeriyor")
    return errs


def build(check_only: bool) -> int:
    errs = [e for s, d in FILES for e in check_file(ROOT / s, d)]
    for e in errs:
        print("HATA", e)
    if errs:
        return 1
    print(f"denetim temiz: {len(FILES)} dosya, görüntü yok, ağırlıklarda yalnız {sorted(ALLOWED_PT_KEYS)}")
    if check_only:
        return 0
    out = ROOT / "dist" / f"guvenloc-{VERSION}-paket"
    if out.exists():
        shutil.rmtree(out)
    for s, d in FILES:
        (out / d).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / s, out / d)
    (out / "README.md").write_text(README.format(v=VERSION), encoding="utf-8")
    shutil.copy2(ROOT / "LICENSE", out / "LICENSE")
    sums = [f"{sha256(out / d)}  {d}" for _, d in FILES]
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    tgz = out.parent / f"{out.name}.tar.gz"
    with tarfile.open(tgz, "w:gz") as t:
        t.add(out, arcname=out.name)
    mb = tgz.stat().st_size / 1e6
    print(f"{tgz.relative_to(ROOT)}  {mb:.1f} MB  sha256 {sha256(tgz)}")
    return 0


def unpack(pkg: Path) -> int:
    pkg = pkg.resolve()
    if pkg.suffix == ".gz":
        dest = ROOT / "dist"
        with tarfile.open(pkg) as t:
            t.extractall(dest, filter="data")
        pkg = dest / f"guvenloc-{VERSION}-paket"
    for line in (pkg / "SHA256SUMS").read_text().splitlines():
        h, d = line.split("  ", 1)
        if sha256(pkg / d) != h:
            print("HATA sha256 uyuşmuyor:", d)
            return 1
    for d, s in UNPACK.items():
        (ROOT / s).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pkg / d, ROOT / s)
        print(f"{d} → {s}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--unpack", type=Path)
    a = ap.parse_args()
    return unpack(a.unpack) if a.unpack else build(a.check)


if __name__ == "__main__":
    sys.exit(main())
