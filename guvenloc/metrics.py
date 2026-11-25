"""Metrikler ve deney kaydı (MIMARI.md §3.2)."""
from __future__ import annotations

import csv
import hashlib
import subprocess
import time
from pathlib import Path

import numpy as np

RUNS_FIELDS = [
    "timestamp", "script", "config_hash", "seed", "git_commit",
    "split", "metric_name", "value", "notes",
]


def config_hash(path: str | Path) -> str:
    """Config dosyasının kısa sha256 özeti."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def log_run(
    script: str,
    config_path: str | Path,
    seed: int,
    split: str,
    metric_name: str,
    value: float,
    notes: str = "",
    runs_csv: str | Path = "results/runs.csv",
) -> None:
    """results/runs.csv dosyasına bir satır ekler (yoksa başlık yazar)."""
    runs_csv = Path(runs_csv)
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not runs_csv.exists()
    with runs_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUNS_FIELDS)
        if new:
            w.writeheader()
        w.writerow({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "script": script,
            "config_hash": config_hash(config_path),
            "seed": seed,
            "git_commit": git_commit(),
            "split": split,
            "metric_name": metric_name,
            "value": value,
            "notes": notes,
        })


# ---------------------------------------------------------------------------
# Değerlendirme metrikleri (MIMARI.md §1F, §4D, §14.2)
# ---------------------------------------------------------------------------


def location_error_m(pred_xy_m, gt_xy_m) -> np.ndarray:
    """Metre cinsinden (N,2) tahmin ve GT konumları arasındaki Öklit hatası."""
    d = np.asarray(pred_xy_m, float) - np.asarray(gt_xy_m, float)
    return np.hypot(d[..., 0], d[..., 1])


def heading_error_deg(pred, gt) -> np.ndarray:
    """Dairesel yön hatası min(|θ̂−φ|, 360−|θ̂−φ|), vektörel."""
    d = np.abs(np.asarray(pred, float) - np.asarray(gt, float)) % 360.0
    return np.minimum(d, 360.0 - d)


def success_at(errors_m, k: float, accepted=None) -> float:
    """Başarı@k: pozitifler içinde kabul edilmiş VE hata ≤ k olanların oranı.

    `accepted` verilmezse (rete izin vermeyen taban çizgileri) tüm kareler kabul sayılır.
    Reddedilen pozitif başarısız sayılır; hatası NaN olan (konum üretilemeyen) da öyle.
    """
    e = np.asarray(errors_m, float)
    acc = np.ones_like(e, bool) if accepted is None else np.asarray(accepted, bool)
    ok = acc & np.isfinite(e) & (e <= k)
    return float(ok.mean()) if e.size else float("nan")


def success_among_accepted(errors_m, k: float, accepted) -> float:
    """Kabul edilenler içinde başarı oranı (ret yapan yöntemlerde ek bilgi)."""
    e = np.asarray(errors_m, float)
    acc = np.asarray(accepted, bool)
    return float((np.isfinite(e[acc]) & (e[acc] <= k)).mean()) if acc.any() else float("nan")


def median_error(errors_m, accepted=None) -> float:
    e = np.asarray(errors_m, float)
    if accepted is not None:
        e = e[np.asarray(accepted, bool)]
    e = e[np.isfinite(e)]
    return float(np.median(e)) if e.size else float("nan")


def auroc(scores, labels) -> float:
    """Mann–Whitney U ile AUROC (bağlar 0,5 sayılır). labels: 1 = pozitif sınıf."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def good_label(is_pos, errors_m, accept_error_m: float = 25.0) -> np.ndarray:
    """AUROC hedefi (§14.2): pozitif ∧ hata ≤ 25 m → 1; negatif ∨ hata > 25 m → 0."""
    e = np.asarray(errors_m, float)
    return np.asarray(is_pos, bool) & np.isfinite(e) & (e <= accept_error_m)


def far_neg(accepted, is_pos) -> float:
    """FAR-neg: negatif çiftlerin kabul edilme oranı (τ'nun kontrol ettiği oran)."""
    neg = ~np.asarray(is_pos, bool)
    return float(np.asarray(accepted, bool)[neg].mean()) if neg.any() else float("nan")


def far_loc(accepted, is_pos, errors_m, accept_error_m: float = 25.0) -> float:
    """FAR-loc: pozitif çiftlerin hata > 25 m ile kabul edilme oranı (pozitifler içinde)."""
    pos = np.asarray(is_pos, bool)
    if not pos.any():
        return float("nan")
    e = np.asarray(errors_m, float)[pos]
    acc = np.asarray(accepted, bool)[pos]
    bad = ~(np.isfinite(e) & (e <= accept_error_m))
    return float((acc & bad).mean())


def coverage(errors_m, radius_m, accepted=None) -> float:
    """Kapsama: kabul edilmiş pozitiflerde GT'nin güven dairesi içinde kalma oranı."""
    e = np.asarray(errors_m, float)
    r = np.asarray(radius_m, float)
    m = np.ones_like(e, bool) if accepted is None else np.asarray(accepted, bool)
    return float((e[m] <= r[m]).mean()) if m.any() else float("nan")


def heading_bin(heading_deg, width: float = 45.0) -> np.ndarray:
    """Yön kırılımı: [0,45), [45,90), ... dilim indisi."""
    return (np.mod(np.asarray(heading_deg, float), 360.0) // width).astype(int)


ALTITUDE_BINS_M = (400.0, 800.0, 1200.0, 2000.0)


def altitude_bin(height_m, edges=ALTITUDE_BINS_M) -> np.ndarray:
    """İrtifa kırılımı: 0 = 400–800, 1 = 800–1200, 2 = 1200–2000; aralık dışı −1."""
    h = np.asarray(height_m, float)
    b = np.digitize(h, edges) - 1
    b[(h < edges[0]) | (h > edges[-1])] = -1
    b[h == edges[-1]] = len(edges) - 2
    return b
