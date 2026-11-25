"""RoMa v2 eşleşme önbelleği (MIMARI.md 1E, INTERFACES.md §1).

Düzen (cache/<sürüm>/):
  <pair_id>_r<rot>.npz            # çift × yön (rot ∈ {0,90,180,270}, drone'a saat yönü)
  seq/<bölge>_<t>_<t+1>.npz       # aynı uçuşta ardışık drone kareleri (Modül 4)
  MANIFEST.csv                    # dosya, pair_id, rot, n, bayt, sha256, setting
Dosya şeması `matcher.MatchSet.save_npz` ile yazılır; burada doğrulanır.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from guvenloc.matcher import NPZ_FIELDS, MatchSet

MANIFEST_FIELDS = ["file", "kind", "pair_id", "rot", "n", "bytes", "sha256", "setting"]
META_KEYS = {"setting", "romav2", "rot_deg", "drone_wh", "sat_wh", "grid_wh", "sigma_scale",
             "bidirectional", "coords", "cov_units"}


def pair_file(cache_dir: str | Path, pair_id: str, rot: int) -> Path:
    return Path(cache_dir) / f"{pair_id}_r{int(rot) % 360}.npz"


def seq_file(cache_dir: str | Path, region: str, t0: int, t1: int) -> Path:
    return Path(cache_dir) / "seq" / f"{region}_{t0:04d}_{t1:04d}.npz"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_npz(path: str | Path) -> list[str]:
    """Şema hataları listesi (boşsa geçerli)."""
    errs: list[str] = []
    try:
        z = np.load(path)
    except Exception as e:  # noqa: BLE001 — bozuk dosya her biçimde raporlanmalı
        return [f"açılamadı: {e!r}"]
    with z:
        missing = (set(NPZ_FIELDS) | {"meta"}) - set(z.files)
        if missing:
            return [f"eksik alan: {sorted(missing)}"]
        u, v, c, SA, SB = (z[k] for k in NPZ_FIELDS)
        n = len(u)
        exp = {"u": ((n, 2), np.float32), "v": ((n, 2), np.float32), "c": ((n,), np.float16),
               "Sigma_A": ((n, 2, 2), np.float32), "Sigma_B": ((n, 2, 2), np.float32)}
        for k, (shape, dt) in exp.items():
            a = z[k]
            if a.shape != shape:
                errs.append(f"{k} şekli {a.shape} ≠ {shape}")
            if a.dtype != dt:
                errs.append(f"{k} tipi {a.dtype} ≠ {np.dtype(dt)}")
            if not np.isfinite(a.astype(np.float32)).all():
                errs.append(f"{k} sonlu olmayan değer içeriyor")
        if errs:
            return errs
        if n == 0:
            errs.append("eşleşme yok")
        if (c < 0).any() or (c.astype(np.float32) > 1.001).any():
            errs.append("c [0,1] dışında")
        for k, S in (("Sigma_A", SA), ("Sigma_B", SB)):
            S = S.astype(np.float32)
            if not np.allclose(S, np.swapaxes(S, 1, 2), atol=1e-6, rtol=1e-5):
                errs.append(f"{k} simetrik değil")
            if (np.linalg.eigvalsh(S.astype(np.float64)) < -1e-6).any():
                errs.append(f"{k} pozitif yarı tanımlı değil")
        try:
            meta = json.loads(str(z["meta"]))
            if META_KEYS - set(meta):
                errs.append(f"meta eksik: {sorted(META_KEYS - set(meta))}")
            else:
                w, h = meta["drone_wh"]
                Ws, Hs = meta["sat_wh"]
                if n and ((u < -1).any() or (u[:, 0] > w).any() or (u[:, 1] > h).any()):
                    errs.append("u drone karesinin dışında")
                if n and ((v < -1).any() or (v[:, 0] > Ws).any() or (v[:, 1] > Hs).any()):
                    errs.append("v karonun dışında")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            errs.append(f"meta okunamadı: {e!r}")
    return errs


def append_manifest(partial: Path, path: Path, kind: str, pair_id: str, rot: int,
                    ms: MatchSet) -> None:
    new = not partial.exists()
    with partial.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"file": path.relative_to(partial.parent).as_posix(), "kind": kind,
                    "pair_id": pair_id, "rot": rot, "n": len(ms), "bytes": path.stat().st_size,
                    "sha256": sha256_file(path), "setting": ms.meta.get("setting", "")})


def partial_name(shard: int = 0, n_shards: int = 1) -> str:
    """Kısmi manifest adı; paralel parçalar (ör. iki GPU) ayrı dosyaya yazar."""
    return "MANIFEST.partial.csv" if n_shards <= 1 else f"MANIFEST.partial.{shard}.csv"


def partial_files(cache_dir: str | Path) -> list[Path]:
    return sorted(Path(cache_dir).glob("MANIFEST.partial*.csv"))


def finalize_manifest(cache_dir: str | Path) -> Path:
    """MANIFEST.partial*.csv'leri birleştirip tekilleştirir, sıralar, eksik dosyaları düşürür
    → MANIFEST.csv."""
    cache_dir = Path(cache_dir)
    rows: dict[str, dict] = {}
    for partial in partial_files(cache_dir):
        with partial.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (cache_dir / r["file"]).exists():
                    rows[r["file"]] = r
    out = cache_dir / "MANIFEST.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    return out
