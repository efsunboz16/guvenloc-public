"""Modül 1 eğitimini (scripts/52_m1_train.py all) canlı izleme paneli — yalnızca okur.

  python scripts/m1_izle.py          # 5 s'de bir yenilenir, Ctrl+C ile çıkılır
  python scripts/m1_izle.py --once
  python scripts/m1_izle.py --ab     # öneri A/B deneyleri (52_m1_train.py ab, logs/m1_ab.log)
  python scripts/m1_izle.py --c      # öneri C fizibilitesi (55_m1c_train.py, logs/m1c.log)
  python scripts/m1_izle.py --cg     # öneri C, gerçek karo (55_m1c_train.py --veri gercek, logs/m1c_gercek.log)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs/m1_train.log"
EPOCHS = 60
PLAN = [("3e-04", 0), ("0.001", 0), ("0.003", 0), ("seçilen", 1), ("seçilen", 2)]
LINE = re.compile(r"lr (\S+) tohum (\d+) epoch (\d+): kayıp ([\d.]+), val medyan hata \(12 açı\) ([\d.]+)°, "
                  r"\(0°\) ([\d.]+)° \[(\d+) s\]")
AB_LINE = re.compile(r"deney (\S+) epoch (\d+): kayıp ([\d.]+), val medyan hata \(12 açı\) ([\d.]+)°, "
                     r"\(0°\) ([\d.]+)° \[(\d+) s\]")
AB_PLAN = ["v1_es", "A1_sert", "A2_sert_gri", "A3_sert_gri_kucuk", "B1_sert_uydu50", "B2_sert_gri_uydu50",
           "B3_yalniz_uydu"]
C_PLAN = ["C1_1e-3", "C2_3e-3", "C3_1e-3_gri"]
AB = {"on": False, "epochs": 40}
TARGET = 10.0
BLOCKS = "▁▂▃▄▅▆▇█"


def lr_key(s: str) -> str:
    return f"{float(s):.0e}".replace("e-0", "e-")


def parse():
    runs: dict[tuple[str, int], list[tuple]] = {}
    chosen, tail, done, summary = None, [], None, []
    if LOG.exists():
        for ln in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
            m = AB_LINE.search(ln) if AB["on"] else None
            if m:
                name, ep, loss, vall, v0, sec = m.groups()
                runs.setdefault((name, 0), []).append((int(ep), float(loss), float(vall), float(v0), int(sec)))
                if ln.strip():
                    tail.append(ln.strip())
                continue
            if AB["on"] and (" bitti:" in ln or ln.startswith("val'e göre")):
                summary.append(ln.strip())
            m = None if AB["on"] else LINE.search(ln)
            if m:
                lr, seed, ep, loss, vall, v0, sec = m.groups()
                runs.setdefault((lr_key(lr), int(seed)), []).append(
                    (int(ep), float(loss), float(vall), float(v0), int(sec)))
            elif ln.startswith("seçilen lr"):
                chosen = ln
            elif ln.startswith("M1_BITTI_KOD"):
                done = ln.split("=")[1]
            elif ln.startswith("  ") and ":" in ln and "±" in ln:
                summary.append(ln.strip())
            if ln.strip():
                tail.append(ln.strip())
    return runs, chosen, tail[-4:], done, summary


def spark(vals: list[float], lo: float = 0, hi: float = 90) -> str:
    out = ""
    for v in vals:
        k = int((min(max(v, lo), hi) - lo) / (hi - lo) * (len(BLOCKS) - 1))
        out += BLOCKS[k]
    return out


def gpu() -> str:
    try:
        q = "--query-gpu=utilization.gpu,memory.used,temperature.gpu"
        u, mu, te = subprocess.run(["nvidia-smi", q, "--format=csv,noheader,nounits"], check=False,
                                   capture_output=True, text=True, timeout=5).stdout.strip().split(", ")
        return f"GPU %{u} · {int(mu) / 1024:.1f} GB · {te} °C"
    except Exception:  # noqa: BLE001
        return "GPU okunamadı"


def alive() -> bool:
    return subprocess.run(["pgrep", "-f", "5[25]_m1c?_train.py"], check=False, capture_output=True).returncode == 0


def render() -> Group:
    runs, chosen, tail, done, summary = parse()
    keys = list(runs)
    plan = [(n, 0) for n in AB_PLAN] if AB["on"] else PLAN
    n_ep = AB["epochs"] if AB["on"] else EPOCHS
    if AB["on"]:
        keys = [k for k in plan if k in runs]
    t = Table(expand=True, box=None, header_style="bold")
    for c in ("koşu", "ilerleme", "son val hata (12 açı)", "0°", "seyir (0–90°)"):
        t.add_column(c)
    for i, (lr, seed) in enumerate(plan):
        if AB["on"]:
            key = (lr, 0) if (lr, 0) in runs else None
            name = lr
        else:
            key = keys[i] if i < len(keys) else None
            name = f"lr {lr} · tohum {seed}" if key is None else f"lr {key[0]} · tohum {key[1]}"
        hist = runs.get(key, []) if key else []
        ep = hist[-1][0] if hist else 0
        bar = Table.grid()
        bar.add_row(ProgressBar(total=n_ep, completed=ep, width=20), f" {ep}/{n_ep}")
        if hist:
            v = hist[-1][2]
            col = "green" if v <= TARGET else ("yellow" if v <= 30 else "red")
            last, last0 = Text(f"{v:.1f}°", style=col), f"{hist[-1][3]:.1f}°"
            sp = Text(spark([h[2] for h in hist]), style=col)
        else:
            last, last0, sp = Text("–", style="dim"), "–", Text("")
        t.add_row(name, bar, last, last0, sp)
    best = None
    allv = [(h[2], k) for k, hs in runs.items() for h in hs]
    if allv:
        best = min(allv)
    status = ("bitti (kod " + done + ")") if done is not None else ("çalışıyor" if alive() else "çalışmıyor")
    head = Text.assemble(("Modül 1 deneyleri  " if AB["on"] else "Modül 1 eğitimi  ", "bold"), ("● " + status, "green" if alive() or done == "0" else "red"),
                         f"   {time.strftime('%H:%M:%S')}   {gpu()}\n",
                         f"Hedef: calib medyan yön hatası ≤ {TARGET:g}°. ",
                         (f"En iyi val şimdiye kadar: {best[0]:.1f}° ({best[1][0]}"
                          + ("" if AB["on"] else f", tohum {best[1][1]}") + ")" if best else ""))
    parts = [Panel(head, border_style="cyan"), Panel(t, title="koşular (doğrulama: bölge 04)", border_style="blue")]
    if chosen:
        parts.append(Panel(chosen, title="öğrenme oranı seçimi", border_style="magenta"))
    if summary:
        parts.append(Panel("\n".join(summary), title="biten deneyler (calib teşhis)" if AB["on"]
                           else "calib sonucu (3 tohum)", border_style="green"))
    parts.append(Panel("\n".join(tail), title="son satırlar", border_style="dim"))
    return Group(*parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--ab", action="store_true", help="A/B deneylerini izle (logs/m1_ab.log)")
    ap.add_argument("--c", action="store_true", help="öneri C fizibilitesini izle (logs/m1c.log)")
    ap.add_argument("--cg", action="store_true", help="öneri C gerçek karo (logs/m1c_gercek.log)")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    global LOG
    if a.cg:
        LOG = ROOT / "logs/m1c_gercek.log"
        AB_PLAN[:] = ["C4_ortalama", "C5_pencere"]
        AB.update(on=True, epochs=a.epochs or 20)
    elif a.c:
        LOG = ROOT / "logs/m1c.log"
        AB_PLAN[:] = C_PLAN
        AB.update(on=True, epochs=a.epochs or 20)
    elif a.ab:
        LOG = ROOT / "logs/m1_ab.log"
        AB.update(on=True, epochs=a.epochs or 40)
    con = Console()
    if a.once:
        con.print(render())
        return
    with Live(render(), console=con, screen=True, auto_refresh=False) as live:
        try:
            while True:
                time.sleep(5)
                live.update(render(), refresh=True)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
