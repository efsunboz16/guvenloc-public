"""Faz 1 koşusunu (scripts/faz1_3090.py) canlı izleme paneli — yalnızca okur, hiçbir şeyi değiştirmez.

Kullanım (ikinci bir terminalde):
  python scripts/faz1_izle.py            # 10 s'de bir yenilenir, Ctrl+C ile çıkılır
  python scripts/faz1_izle.py --once     # tek seferlik görüntü
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs/faz1_3090.log"
CACHE = ROOT / "cache/v1"
PAIRS = ROOT / "data/pairs/pairs_v1.csv"
RES = ROOT / "results/faz1"
STEPS = ["env", "data", "pairs", "speed", "megadepth", "smoke", "cache_dev", "baselines", "repro",
         "cache_rest"]
SEEDS = 5
N_ROT = 4
TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (.*)$")


def expected_counts() -> tuple[dict[str, str], Counter, int]:
    """pair_id → split, split başına beklenen dosya sayısı, beklenen seq dosyası sayısı."""
    split_of, exp, pos = {}, Counter(), Counter()
    with PAIRS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            split_of[r["pair_id"]] = r["split"]
            exp[r["split"]] += N_ROT
            if r["neg_type"] == "pos":
                pos[r["region"]] += 1
    return split_of, exp, sum(max(n - 1, 0) for n in pos.values())


def parse_log() -> tuple[dict[str, dict], str | None, list[str]]:
    st: dict[str, dict] = {}
    cur, tail = None, []
    if not LOG.exists():
        return st, cur, tail
    for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        m = TS.match(line)
        if not m:
            continue
        t, msg = datetime.strptime(m[1], "%Y-%m-%d %H:%M:%S"), m[2]
        if msg.startswith("=== ") and " bitti" in msg:
            name = msg[4:].split()[0]
            st.setdefault(name, {})["end"] = t
            st[name].pop("fail", None)
            cur = None
        elif msg.startswith("=== "):
            name = msg[4:].split(":")[0]
            st[name] = {"start": t}
            cur = name
        elif msg.startswith("!!! "):
            name = msg[4:].split()[0]
            st.setdefault(name, {})["fail"] = msg
            cur = None
        tail.append(f"{m[1][11:]} {msg}")
    return st, cur, tail[-6:]


def running() -> bool:
    p = subprocess.run(["pgrep", "-f", "scripts/faz1_3090.py"], check=False, capture_output=True)
    return p.returncode == 0


def gpu() -> str:
    try:
        q = "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
        out = subprocess.run(["nvidia-smi", q, "--format=csv,noheader,nounits"], check=False,
                             capture_output=True, text=True, timeout=5).stdout.strip().split(", ")
        u, mu, mt, te, pw = out
        return f"%{u} yük · {int(mu) / 1024:.1f}/{int(mt) / 1024:.0f} GB · {te} °C · {float(pw):.0f} W"
    except Exception:  # noqa: BLE001
        return "okunamadı"


def bar(done: int, total: int, rate: float | None) -> Table:
    t = Table.grid(padding=(0, 1))
    t.add_column(width=34)
    t.add_column()
    frac = done / total if total else 0
    eta = ""
    if rate and done < total:
        eta = f" · kalan ≈ {(total - done) / rate / 3600:.1f} sa"
    t.add_row(ProgressBar(total=max(total, 1), completed=done, width=34),
              f"{done:>6}/{total} (%{100 * frac:.0f}){eta}")
    return t


class Rate:
    """Son birkaç ölçümden dosya/s."""

    def __init__(self):
        self.hist: list[tuple[float, int]] = []

    def update(self, n: int) -> float | None:
        now = time.time()
        self.hist.append((now, n))
        self.hist = [(t, k) for t, k in self.hist if now - t < 600]
        (t0, k0), (t1, k1) = self.hist[0], self.hist[-1]
        return (k1 - k0) / (t1 - t0) if t1 - t0 > 20 and k1 > k0 else None


def render(exp_info, rate: Rate) -> Group:
    split_of, exp, exp_seq = exp_info
    st, cur, tail = parse_log()
    alive = running()

    # adımlar
    steps = Table(show_header=True, header_style="bold", expand=True, box=None)
    steps.add_column("#", width=3)
    steps.add_column("adım")
    steps.add_column("durum")
    steps.add_column("süre", justify="right")
    now = datetime.now()
    for i, s in enumerate(STEPS, 1):
        d = st.get(s, {})
        if "fail" in d:
            state, dur = Text("✗ durdu", style="bold red"), ""
        elif "end" in d:
            state = Text("✓ bitti", style="green")
            dur = d["end"] - d["start"] if "start" in d else None
        elif s == cur:
            state = Text("▶ çalışıyor" if alive else "■ durdu (süreç yok)",
                         style="bold yellow" if alive else "bold red")
            dur = now - d["start"]
        else:
            state, dur = Text("· bekliyor", style="dim"), None
        ds = f"{int(dur.total_seconds() // 3600)} sa {int(dur.total_seconds() % 3600 // 60):02d} dk" \
            if dur else ""
        steps.add_row(str(i), s, state, ds)

    # önbellek
    have = Counter()
    n_seq = 0
    if CACHE.exists():
        for f in CACHE.glob("*.npz"):
            if f.name.endswith(".tmp.npz"):
                continue
            have[split_of.get(f.stem.rsplit("_r", 1)[0], "?")] += 1
        seq_dir = CACHE / "seq"
        n_seq = sum(1 for f in seq_dir.glob("*.npz") if not f.name.endswith(".tmp.npz")) \
            if seq_dir.exists() else 0
    total_done = sum(have.values()) + n_seq
    r = rate.update(total_done)
    cache = Table.grid(padding=(0, 2))
    cache.add_column(width=8)
    cache.add_column()
    for sp in ("calib", "train", "seq", "test"):
        done, tot = (n_seq, exp_seq) if sp == "seq" else (have[sp], exp[sp])
        active = (done < tot) and r is not None and (
            (cur == "cache_dev" and sp in ("calib", "train")) or (cur == "cache_rest" and sp in ("seq", "test")))
        cache.add_row(sp, bar(done, tot, r if active else None))
    speed = f"{r:.1f} dosya/s" if r else "hesaplanıyor… (≈ 30 s)"
    size = sum(f.stat().st_size for f in CACHE.rglob("*.npz")) / 1e9 if CACHE.exists() else 0

    # tekrarlama
    rep = Table.grid(padding=(0, 2))
    rep.add_column(width=10)
    rep.add_column()
    for name, label in (("mega1500", "MegaDepth"), ("c4", "C4")):
        n = len(list(RES.glob(f"repro_{name}_seed*.json"))) if RES.exists() else 0
        rep.add_row(label, bar(n, SEEDS, None))

    du = shutil.disk_usage(ROOT)
    head = Text.assemble(
        ("Faz 1 koşusu  ", "bold"),
        ("● çalışıyor" if alive else "● çalışmıyor", "green" if alive else "red"),
        f"   {now:%H:%M:%S}\n",
        f"GPU: {gpu()}\nDisk: {du.free / 1e9:.0f} GB boş · önbellek {size:.1f} GB · hız {speed}")
    return Group(
        Panel(head, border_style="cyan"),
        Panel(steps, title="adımlar", border_style="blue"),
        Panel(cache, title="önbellek cache/v1", border_style="blue"),
        Panel(rep, title="tekrarlama (tohum)", border_style="blue"),
        Panel("\n".join(tail), title="son günlük satırları", border_style="dim"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", type=float, default=10.0)
    a = ap.parse_args()
    exp_info = expected_counts()
    rate = Rate()
    con = Console()
    if a.once:
        con.print(render(exp_info, rate))
        return
    with Live(render(exp_info, rate), console=con, screen=True, auto_refresh=False) as live:
        try:
            while True:
                time.sleep(a.every)
                live.update(render(exp_info, rate), refresh=True)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
