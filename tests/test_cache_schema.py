"""Önbellek şema testi (MIMARI.md 1E): cache/v1 varsa rastgele 100 dosya doğrulanır.

Önbellek yoksa (CI, önbelleği olmayan makine) atlanır. Şema mantığı ayrıca sentetik
dosyalarla test edilir (her ortamda koşar).
"""
import json
import random
from pathlib import Path

import numpy as np
import pytest

from guvenloc.cache import finalize_manifest, validate_npz
from guvenloc.matcher import MatchSet

CACHE = Path(__file__).resolve().parents[1] / "cache" / "v1"
META = {"setting": "base", "romav2": "2.0.1", "rot_deg": 0.0, "drone_wh": [100, 80],
        "sat_wh": [64, 64], "grid_wh": [640, 640], "sigma_scale": 1.0, "bidirectional": False,
        "coords": "opencv", "cov_units": "px2"}


def _ms(n=20, **kw):
    rng = np.random.default_rng(0)
    d = {"u": rng.uniform(0, 79, (n, 2)).astype(np.float32),
             "v": rng.uniform(0, 63, (n, 2)).astype(np.float32),
             "c": rng.uniform(0, 1, n).astype(np.float32),
             "Sigma_A": np.zeros((n, 2, 2), np.float32),
             "Sigma_B": np.tile(np.eye(2, dtype=np.float32), (n, 1, 1)), "meta": dict(META)}
    d.update(kw)
    return MatchSet(**d)


def test_validate_accepts_good_and_rejects_bad(tmp_path):
    good = tmp_path / "good.npz"
    _ms().save_npz(good)
    assert validate_npz(good) == []

    bad = tmp_path / "bad.npz"
    np.savez(bad, u=np.zeros((3, 2), np.float32), meta=json.dumps(META))
    assert any("eksik alan" in e for e in validate_npz(bad))

    out = tmp_path / "out.npz"
    _ms(v=np.full((20, 2), 500, np.float32)).save_npz(out)
    assert any("karonun dışında" in e for e in validate_npz(out))

    (tmp_path / "broken.npz").write_bytes(b"not a zip")
    assert any("açılamadı" in e for e in validate_npz(tmp_path / "broken.npz"))


def test_manifest_dedup_and_missing(tmp_path):
    p = tmp_path / "MANIFEST.partial.csv"
    p.write_text("file,kind,pair_id,rot,n,bytes,sha256,setting\n"
                 "b.npz,pair,b,0,1,1,x,base\na.npz,pair,a,0,1,1,old,base\n"
                 "a.npz,pair,a,0,1,1,new,base\ngone.npz,pair,g,0,1,1,x,base\n", encoding="utf-8")
    (tmp_path / "a.npz").write_bytes(b"1")
    (tmp_path / "b.npz").write_bytes(b"1")
    rows = finalize_manifest(tmp_path).read_text(encoding="utf-8").splitlines()
    assert rows[1].startswith("a.npz") and ",new," in rows[1]
    assert len(rows) == 3   # başlık + a + b (gone.npz düştü)


@pytest.mark.skipif(not CACHE.exists(), reason="cache/v1 yok")
def test_real_cache_random_100():
    files = sorted(CACHE.glob("*.npz")) + sorted((CACHE / "seq").glob("*.npz"))
    if not files:
        pytest.skip("cache/v1 boş")
    sample = random.Random(0).sample(files, min(100, len(files)))
    bad = {f.name: e for f in sample if (e := validate_npz(f))}
    assert not bad, bad
