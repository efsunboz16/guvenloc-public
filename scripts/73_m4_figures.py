"""Faz 5D – görseller: GT yörünge, tek kare tahminleri, filtre yörüngesi, parçacık bulutları, γ ödünleşimi.

  python scripts/73_m4_figures.py          # calib (05, 11) + bölge özetleri; test bölgeleri Faz 6'da
→ results/faz5/*.png (kopyaları docs/sonuclar/faz5/sekiller/)

Renkler: filtre mavi (#257de0), tek kare turuncu (#f47230, M3 reddi soluk), GT gri kesikli.
Ayar tune.json, κ h4_dry.json, γ gamma.json; yeni hesap yok, 72_m4_filter.run yeniden koşulur (tohum 0).
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import yaml

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from guvenloc import m4_filter as F


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S72 = _load("m4s72", "scripts/72_m4_filter.py")
R5, OUT, DOCS = Path("results/faz5/m4"), Path("results/faz5"), Path("docs/sonuclar/faz5/sekiller")
C_F, C_S, C_G, INK, MUTE = "#257de0", "#f47230", "#8a8a84", "#191919", "#666666"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#cfcdc6", "axes.labelcolor": INK, "xtick.color": MUTE,
                     "ytick.color": MUTE, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
                     "axes.grid": True, "grid.color": "#ecebe6", "grid.linewidth": 0.8, "legend.frameon": False})


def traj_fig(df: pd.DataFrame, rid: str, win: tuple[int, int]) -> Path:
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, sub, title in ((axs[0], df, f"{rid}: bütün uçuş ({len(df)} kare)"),
                           (axs[1], df[df.num.between(*win)], f"{rid}: kare {win[0]}–{win[1]}")):
        lim = sub[["ge", "gn", "e7", "n7"]]
        pad = 80
        x0, x1 = min(lim["ge"].min(), lim.e7.min()) - pad, max(lim["ge"].max(), lim.e7.max()) + pad
        y0, y1 = min(lim["gn"].min(), lim.n7.min()) - pad, max(lim["gn"].max(), lim.n7.max()) + pad
        acc = sub.accept.to_numpy()
        ax.scatter(sub.e6.clip(x0, x1)[acc], sub.n6.clip(y0, y1)[acc], s=9, color=C_S, alpha=0.85, lw=0,
                   label="tek kare, M3 kabul")
        ax.scatter(sub.e6.clip(x0, x1)[~acc], sub.n6.clip(y0, y1)[~acc], s=9, color=C_S, alpha=0.3, lw=0,
                   label="tek kare, M3 ret")
        ax.plot(sub["ge"], sub["gn"], "--", color=C_G, lw=1.6, label="GT (GPS)")
        ax.plot(sub.e7, sub.n7, color=C_F, lw=1.8, label="filtre")
        rs = sub[sub.reset]
        ax.scatter(rs.e7, rs.n7, marker="D", s=26, facecolor="white", edgecolor=INK, lw=1, zorder=5, label="sıfırlama")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("doğu (m)")
        ax.set_ylabel("kuzey (m)")
    axs[0].legend(loc="upper left", fontsize=8.5, ncol=1)
    m6, m7 = df.err6.median(), df.err7.median()
    fig.suptitle(f"Faz 5 — bölge {rid} (calib): medyan hata tek kare {m6:.0f} m → filtre {m7:.0f} m "
                 f"(haritanın dışına düşen tek kare tepeleri kenara kırpıldı)", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    p = OUT / f"traj_{rid}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def err_fig(dfs: dict) -> Path:
    fig, axs = plt.subplots(len(dfs), 1, figsize=(12, 3.2 * len(dfs)), sharey=True)
    for ax, (rid, df) in zip(np.atleast_1d(axs), dfs.items(), strict=True):
        rej = ~df.accept.to_numpy()
        for i in np.flatnonzero(rej):
            ax.axvspan(df.num.iloc[i] - 0.5, df.num.iloc[i] + 0.5, color="#ecebe6", lw=0)
        ax.scatter(df.num, df.err6.clip(3, 5000), s=7, color=C_S, alpha=0.8, lw=0, label="tek kare")
        ax.plot(df.num, df.err7.clip(3, 5000), color=C_F, lw=1.4, label="filtre")
        for n in df.num[df.reset]:
            ax.plot([n, n], [3, 4.2], color=INK, lw=1)
        ax.set_yscale("log")
        ax.set_ylim(3, 5000)
        ax.set_ylabel("konum hatası (m)")
        ax.set_title(f"{rid}: gri = M3 reddi (güncelleme yok), alttaki çentikler = sıfırlama", loc="left", fontsize=10.5)
    axs[-1].set_xlabel("kare")
    axs[0].legend(loc="upper right", fontsize=9, ncol=2)
    fig.tight_layout()
    p = OUT / "err_calib.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def pick_snaps(df: pd.DataFrame) -> list[int]:
    """Üç örnek kare: (1) uzun bir M3 ret dizisinin sonu, (2) bir sıfırlamadan hemen önce, (3) iyi izleme."""
    acc = df.accept.to_numpy()
    best, cur, end = 0, 0, None
    for i, a in enumerate(acc):
        cur = 0 if a else cur + 1
        if cur > best:
            best, end = cur, i
    out = [int(df.num.iloc[end])] if end is not None else []
    rs = df.index[df.reset & (df.num > 30)]
    if len(rs):
        out.append(int(df.num.loc[rs[0] - 1]))
    prev_reset = df.reset.shift(1, fill_value=False)
    good = df[(df.err7 < 15) & df.accept & (df.num > 60) & ~df.reset & ~prev_reset & (df.spread < 40)]
    if len(good):
        out.append(int(good.num.iloc[len(good) // 2]))
    return out


def cloud_fig(R: dict, cfg: F.FilterCfg, df: pd.DataFrame, rid: str) -> Path:
    want = pick_snaps(df)
    snaps = dict.fromkeys(want)
    S72.run(R, cfg, "7", 0.0, snaps)
    kappa = json.loads((R5 / "h4_dry.json").read_text(encoding="utf-8"))["7"]["pooled_in_sample"]["kappa"]
    fig, axs = plt.subplots(1, len(want), figsize=(4.3 * len(want), 4.6))
    labels = ["uzun M3 ret dizisinin sonu", "sıfırlamadan bir önceki kare", "iyi izleme"]
    for ax, num, lab in zip(np.atleast_1d(axs), want, labels, strict=False):
        x, w = snaps[num]
        row = df[df.num == num].iloc[0]
        k = int(np.flatnonzero(R["d"].num.to_numpy() == num)[0])
        P = R["meas"][k]
        e0, n0 = row["ge"], row["gn"]
        half = max(250.0, 1.3 * float(np.hypot(row.e7 - e0, row.n7 - n0)) + 60)
        ax.scatter(x[:, 0], x[:, 1], s=4, c=C_F, alpha=np.clip(w / w.max(), 0.08, 1), lw=0, label="parçacıklar")
        ax.add_patch(plt.Circle((row.e7, row.n7), kappa * row.spread, fill=False, color=C_F, lw=1.2, ls=":",
                                label=f"yarıçap κ·yayılım = {kappa * row.spread:.0f} m"))
        ax.plot(row.e7, row.n7, "o", color=C_F, ms=7, mec="white", label="filtre")
        ax.plot(np.clip(row.e6, e0 - half, e0 + half), np.clip(row.n6, n0 - half, n0 + half), "o", color=C_S, ms=7,
                mec="white", alpha=1 if row.accept else 0.45, label="tek kare" + ("" if row.accept else " (ret)"))
        ax.plot(e0, n0, "x", color=INK, ms=9, mew=2, label="GT")
        cx, cy = P.center_en
        tm = P.tile_px * P.mpp / 2
        ax.add_patch(plt.Rectangle((cx - tm, cy - tm), 2 * tm, 2 * tm, fill=False, color=C_G, lw=1, ls="--"))
        ax.set_xlim(e0 - half, e0 + half)
        ax.set_ylim(n0 - half, n0 + half)
        ax.set_aspect("equal")
        ax.set_title(f"{rid} kare {num}: {lab}\nhata tek kare {row.err6:.0f} m · filtre {row.err7:.0f} m",
                     loc="left", fontsize=9.5)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Parçacık bulutları (bölge {rid}); kesikli kare = o karenin uydu karosu", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout()
    p = OUT / f"clouds_{rid}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def gamma_fig() -> Path:
    g = json.loads((R5 / "gamma.json").read_text(encoding="utf-8"))
    d = pd.DataFrame(g["grid"])
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(d.skip_rate, d.med7, "-o", color=C_F, lw=2, ms=6)
    for r in d.itertuples():
        off = (6, 6) if r.gamma == 25 else (6, -12)
        ax.annotate(f"γ = {r.gamma}", (r.skip_rate, r.med7), textcoords="offset points", xytext=off, fontsize=8,
                    color=MUTE)
    s = d[d.gamma == g["selected"]].iloc[0]
    ax.plot(s.skip_rate, s.med7, "o", ms=13, mfc="none", mec=INK, mew=1.5)
    ax.axhline(1.05 * d.med7.iloc[0], color=MUTE, ls=":", lw=1)
    ax.text(0.01, 1.05 * d.med7.iloc[0] + 3, "1,05 × her karede eşleştirme (kural sınırı)", fontsize=8, color=MUTE)
    ax.axhline(d.med6.iloc[0], color=C_S, ls="--", lw=1.2)
    ax.text(0.01, d.med6.iloc[0] + 3, f"tek kare medyanı {d.med6.iloc[0]:.0f} m", fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_yticks([30, 40, 60, 100, 150, 200, 300], ["30", "40", "60", "100", "150", "200", "300"])
    ax.minorticks_off()
    ax.set_xlabel("uydu eşleştirmesi atlanan karelerin oranı")
    ax.set_ylabel("medyan konum hatası (m)")
    ax.set_title(f"5C — hesap/hata ödünleşimi (calib 05 + 11); seçilen γ = {g['selected']} m", loc="left", fontsize=11)
    fig.tight_layout()
    p = OUT / "gamma.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def regions_fig() -> Path:
    t = pd.read_csv(R5 / "traj.csv", dtype={"region": str})
    m = t.groupby("region")[["err6", "err7"]].median()
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(m))
    ax.bar(x - 0.2, m.err6, 0.38, color=C_S, label="tek kare (satır 6)")
    ax.bar(x + 0.2, m.err7, 0.38, color=C_F, label="filtre (satır 7)")
    for i, (a, b) in enumerate(zip(m.err6, m.err7, strict=True)):
        ax.text(i - 0.2, a * 1.08, f"{a:.0f}", ha="center", fontsize=8.5, color=INK)
        ax.text(i + 0.2, b * 1.08, f"{b:.0f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, [f"{r}\n{'calib' if r in ('05', '11') else 'eğitim'}" for r in m.index])
    ax.set_yscale("log")
    ax.set_ylabel("medyan konum hatası (m)")
    ax.set_title("Bölge başına medyan hata (eğitim bölgeleri ayarda kullanıldı: örneklem içi)", loc="left", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = OUT / "regions.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    h4 = yaml.safe_load(Path("configs/h4.yaml").read_text(encoding="utf-8"))
    params = json.loads(Path("results/faz4/m3/m3_params_h3_v1.json").read_text(encoding="utf-8"))
    cfg = F.FilterCfg(**json.loads((R5 / "tune.json").read_text(encoding="utf-8"))["selected"])
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    paths, dfs = [], {}
    for rid in h4["calibration_regions"]:
        R = S72.load_region(rid, params)
        df = S72.run(R, cfg)
        dfs[rid] = df
        n = len(df)
        a = int(df.num.iloc[n // 3])
        paths += [traj_fig(df, rid, (a, a + 60)), cloud_fig(R, cfg, df, rid)]
    paths += [err_fig(dfs), gamma_fig(), regions_fig()]
    for p in paths:
        shutil.copy(p, DOCS / p.name)
        print("yazıldı:", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
