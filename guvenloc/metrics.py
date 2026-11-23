"""Metrikler ve deney kaydı (MIMARI.md §3.2)."""
from __future__ import annotations

import csv
import hashlib
import subprocess
import time
from pathlib import Path

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
