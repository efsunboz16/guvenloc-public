"""Faz 1E – önbelleğin TÜM dosyalarını şema doğrulamasından geçir (çökme/kopya sonrası).

Kullanım:
  python scripts/31_verify_cache.py cache/dev_base               # yalnızca rapor
  python scripts/31_verify_cache.py cache/dev_base --delete-bad  # bozukları sil (30_build_cache
                                                                 # bir sonraki koşuda yeniden üretir)
Ayrıca yarım kalmış *.tmp.npz dosyalarını temizler, manifest'te kaydı olmayan geçerli
dosyaları ekler ve MANIFEST.csv'yi yeniden üretir.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc.cache import append_manifest, finalize_manifest, partial_files, validate_npz
from guvenloc.matcher import MatchSet


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path)
    ap.add_argument("--delete-bad", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    tmp = list(a.cache.rglob("*.tmp.npz"))
    for t in tmp:
        t.unlink()
    files = sorted(p for p in a.cache.rglob("*.npz"))
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(validate_npz, files, chunksize=64))
    bad = [(f, e) for f, e in zip(files, res, strict=True) if e]
    print(f"{len(files)} dosya, bozuk {len(bad)}, silinen yarım .tmp {len(tmp)}")
    for f, e in bad[:20]:
        print(f"  {f.name}: {e}")
    if bad and a.delete_bad:
        for f, _ in bad:
            f.unlink()
        print(f"{len(bad)} bozuk dosya silindi")
    # manifest'te kaydı olmayan geçerli dosyaları ekle (çökme anında satır yazılamamış olabilir)
    import csv

    partial = a.cache / "MANIFEST.partial.csv"
    known = set()
    for pf in partial_files(a.cache):
        with pf.open(encoding="utf-8") as f:
            known |= {r["file"] for r in csv.DictReader(f)}
    bad_set = {f for f, _ in bad}
    added = 0
    for f in files:
        rel = f.relative_to(a.cache).as_posix()
        if rel in known or f in bad_set or not f.exists():
            continue
        ms = MatchSet.load_npz(f)
        kind = ms.meta.get("kind", "seq" if f.parent.name == "seq" else "pair")
        pid = ms.meta.get("pair_id", f.stem.rsplit("_r", 1)[0])
        append_manifest(partial, f, kind, pid, int(ms.meta.get("rot_deg", 0)), ms)
        added += 1
    print(f"manifest'e eklenen eksik kayıt: {added}")
    print("manifest:", finalize_manifest(a.cache))
    return 1 if bad and not a.delete_bad else 0


if __name__ == "__main__":
    sys.exit(main())
