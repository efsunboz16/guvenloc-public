"""Faz 2E – M1-C değerlendirmesi: (1) eğitilmiş ağda ara açı eşdeğerliği, (2) hız ve kare başına süre.

  python scripts/57_m1c_2e.py                          # F2-9 modeli + 3 tohum eşdeğerlik
  python scripts/57_m1c_2e.py --models D2_olcek_rot90_s2

(1) Calib pozitifleri (drone karesi + sistemdeki gerçek karo, relsat_real_513 'eval'): drone saat yönünde α
    döndürülünce tahmin −α kaymalı. Sapma = wrap(φ̂(α) − (φ̂(0) − α)). 90°'nin katlarında yapı gereği ≈ 0
    (2B), 45°'nin tek katları ve ara açılarda bilineer yeniden örnekleme nedeniyle yaklaşık.
(2) 3090, tek kare (grup 1), uint8 GPU'da → ön işlem + iki gövde + korelasyon; ısınma sonrası medyan.
    Kare başına beklenen süre = M1-C + çağrı × RoMa `base` (results/faz1/speed.json, matcher_s);
    çağrı sayıları 56_m1c_roma.py calib özetinden (roma_row4_<model>_calib.json).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from guvenloc import m1_canon as M
from guvenloc.metrics import log_run


def _load(name, file):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C55 = _load("m1c55", "scripts/55_m1c_train.py")
OUT = Path("results/faz2/m1")
ANGLES = [0.0, 10.0, 20.0, 30.0, 45.0, 90.0, 135.0, 180.0, 200.0, 270.0, 315.0]


def build(model: str):
    import re
    base = re.sub(r"_s\d+$", "", model)
    _lr, cfg, pool = (C55.CONFIGS_GERCEK[base] if base in C55.CONFIGS_GERCEK else C55.CONFIGS_V2[base][:3])
    net = M.C8RelRot(sat_pool=pool).cuda()
    net.load_state_dict(torch.load(f"cache/models/m1c_{model}.pt", map_location="cuda")["state_dict"])
    return net.eval(), cfg


def wrap(x):
    return (x + 180.0) % 360.0 - 180.0


@torch.no_grad()
def equivariance(net, cfg, D) -> dict:
    _, _, frames, sat, phi, lidx, rows = D
    j = rows["eval"]
    preds = {a: [] for a in ANGLES}
    for i in range(0, len(j), 64):
        jj = j[i:i + 64]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ds = net.describe_sat(C55.make_sat(C55.get_sat(sat, jj), False, cfg))
            for a in ANGLES:
                al = torch.full((len(jj),), a, device=frames.device)
                lg = net.score(net.describe(C55.T.make_batch(frames[lidx[jj]], al, False, cfg)), ds).float()
                preds[a].append(M.predict_deg(lg))
    p = {a: torch.cat(v).cpu().numpy() for a, v in preds.items()}
    gt = phi[lidx[j]].cpu().numpy()
    out = {}
    for a in ANGLES:
        dev = np.abs(wrap(p[a] - (p[0.0] - a)))
        err = np.abs(wrap(p[a] - (gt - a)))
        out[str(a)] = {"dev_median": float(np.median(dev)), "dev_p90": float(np.percentile(dev, 90)),
                       "dev_gt5": float((dev > 5).mean()), "dev_gt45": float((dev > 45).mean()),
                       "err_median": float(np.median(err)), "acc45": float((err <= 45).mean())}
    return out


def timed(fn, reps=100, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()          # GPU'da duran veri/ağırlık hariç: yalnız çıkarımın ek belleği
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), float(np.percentile(ts, 90)), (torch.cuda.max_memory_allocated() - base) / 1e9


@torch.no_grad()
def speed(net, cfg) -> dict:
    d = torch.randint(0, 255, (1, 321, 321, 3), dtype=torch.uint8, device="cuda")
    s = torch.randint(0, 255, (1, 513, 513, 3), dtype=torch.uint8, device="cuda")
    z = torch.zeros(1, device="cuda")
    res = {}
    for prec in ("bf16", "fp32"):
        ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if prec == "bf16" else \
              (lambda: torch.autocast("cuda", enabled=False))

        def full(ctx=ctx):
            with ctx():
                lg = net.score(net.describe(C55.T.make_batch(d, z, False, cfg)),
                               net.describe_sat(C55.make_sat(s, False, cfg))).float()
                return M.predict_deg(lg)

        def sat_only(ctx=ctx):
            with ctx():
                return net.describe_sat(C55.make_sat(s, False, cfg))

        def drone_only(ctx=ctx):
            with ctx():
                return net.describe(C55.T.make_batch(d, z, False, cfg))

        t, t90, gb = timed(full)
        res[prec] = {"weights_gb": sum(p.numel() * p.element_size() for p in net.parameters()) / 1e9, "m1c_s": t, "m1c_p90_s": t90, "peak_gb": gb,
                     "sat_s": timed(sat_only)[0], "drone_s": timed(drone_only)[0]}
    return res


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="D2_olcek_rot90_s2", help="v0.1 modeli (F2-9)")
    ap.add_argument("--models", nargs="+", default=["D2_olcek_rot90", "D2_olcek_rot90_s1", "D2_olcek_rot90_s2"])
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    C55.MODE["veri"] = "gercek"
    D = C55.load("cuda")

    eq = {}
    for m in a.models:
        net, cfg = build(m)
        eq[m] = equivariance(net, cfg, D)
        del net
    net, cfg = build(a.main)
    sp = speed(net, cfg)
    roma = {r["setting"]: r for r in json.loads(Path("results/faz1/speed.json").read_text())["results"]}["base"]
    t_roma = roma["matcher_s"]
    summ = json.loads((OUT / f"roma_row4_{a.main}_calib.json").read_text(encoding="utf-8"))["summary"]
    t_m1 = sp["bf16"]["m1c_s"]
    cost = {}
    for row in ("1", "2", "4a", "4c", "4d", "4e"):
        m1 = t_m1 if row.startswith("4") else 0.0
        cost[row] = {k: m1 + summ[row][f"calls_mean{suf}"] * t_roma
                     for k, suf in (("pos_s", "_pos"), ("neg_s", "_neg"), ("all_s", ""))}
        cost[row]["success@25"] = summ[row]["success@25"]

    lines = [f"# Faz 2E — M1-C eşdeğerlik ve hız (v0.1 = `{a.main}`)", "",
             "## Ara açı eşdeğerliği (calib pozitifleri, drone saat yönünde α; sapma = |φ̂(α) − (φ̂(0) − α)|)", "",
             "| α | " + " | ".join(f"{m}: medyan / p90 / > 5° / > 45°" for m in a.models) + " | yön hatası medyanı (v0.1) |",
             "|---|" + "---|" * (len(a.models) + 1)]
    for ang in ANGLES[1:]:
        k = str(ang)
        cells = [f"{eq[m][k]['dev_median']:.2f}° / {eq[m][k]['dev_p90']:.1f}° / %{100 * eq[m][k]['dev_gt5']:.0f} / "
                 f"%{100 * eq[m][k]['dev_gt45']:.1f}" for m in a.models]
        lines.append(f"| {ang:g}° | " + " | ".join(cells) + f" | {eq[a.main][k]['err_median']:.1f}° |")
    lines += ["", f"## Hız (RTX 3090, tek kare, {roma['setting']} RoMa = {t_roma * 1000:.0f} ms/çağrı)", "",
              "| hassasiyet | M1-C toplam | karo gövdesi | kare gövdesi | p90 | ek tepe bellek |", "|---|---|---|---|---|---|"]
    for prec, r in sp.items():
        lines.append(f"| {prec} | {r['m1c_s'] * 1000:.1f} ms | {r['sat_s'] * 1000:.1f} ms | {r['drone_s'] * 1000:.1f} ms | "
                     f"{r['m1c_p90_s'] * 1000:.1f} ms | {r['peak_gb']:.2f} GB |")
    lines += ["", "## Kare başına beklenen süre (calib çağrı sayıları; M1-C bf16 dahil)", "",
              "| satır | başarı@25 | pozitif kare | negatif kare | tümü | satır 2'ye göre (pozitif) |", "|---|---|---|---|---|---|"]
    for row, c in cost.items():
        lines.append(f"| {row} | {c['success@25']:.3f} | {c['pos_s'] * 1000:.0f} ms | {c['neg_s'] * 1000:.0f} ms | "
                     f"{c['all_s'] * 1000:.0f} ms | ×{cost['2']['pos_s'] / c['pos_s']:.2f} |")
    md = OUT / "m1c_2e.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "m1c_2e.json").write_text(json.dumps({"equivariance": eq, "speed": sp, "roma_base_s": t_roma,
                                                 "cost": cost, "main": a.main}, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    print("\n".join(lines))
    log_run("57_m1c_2e.py", Path("configs/h1.yaml"), 0, "calib", "m1c_ms_bf16", round(t_m1 * 1000, 3),
            notes=f"model={a.main}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
