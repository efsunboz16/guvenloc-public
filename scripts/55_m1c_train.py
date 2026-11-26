"""Faz 2C (öneri C fizibilitesi) – göreli dönme modeli (M.C8RelRot): drone karesi + kuzey-yukarı uydu karosu → φ.

  python scripts/55_m1c_train.py                     # 3 yapılandırma, tohum 0 → results/faz2/m1/m1c.json
  python scripts/55_m1c_train.py --only C1_1e-3

Veri: data/m1/frames_321.npy + relsat_321.npy (scripts/54_m1c_prep.py). Eğitim yalnız 01/08/09/10;
seçim val 04 (12 açı medyanı, erken durdurma), calib 05/11 teşhis. Drone karesi U(0, 360°) döndürülür,
hedef φ − α; uydu karosu döndürülmez. Ön işleme/oynatma 52_m1_train.make_batch ile aynı.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from guvenloc import m1_canon as M

_spec = importlib.util.spec_from_file_location("m1_train", ROOT / "scripts/52_m1_train.py")
T = importlib.util.module_from_spec(_spec)
sys.modules["m1_train"] = T
_spec.loader.exec_module(T)

OUT = Path("results/faz2/m1")
BATCH = 64
CONFIGS = {
    "C1_1e-3": (1e-3, T.Cfg("C1", aug="strong", early_stop=True)),
    "C2_3e-3": (3e-3, T.Cfg("C2", aug="strong", early_stop=True)),
    "C3_1e-3_gri": (1e-3, T.Cfg("C3", aug="strong", gray=True, early_stop=True)),
}


def load(dev):
    lab = pd.read_csv("data/m1/labels_v2.csv", dtype={"region": str})
    rs = pd.read_csv("data/m1/relsat_321.csv", dtype={"region": str})
    frames = torch.from_numpy(np.load("data/m1/frames_321.npy")).to(dev)
    sat = torch.from_numpy(np.load("data/m1/relsat_321.npy")).to(dev)
    phi = torch.tensor(lab.phi_corr.to_numpy(), dtype=torch.float32, device=dev)
    lidx = torch.tensor(rs.lab_idx.to_numpy(), device=dev)
    rows = {r: torch.tensor(np.flatnonzero(rs.m1_role.to_numpy() == r), device=dev) for r in ("train", "val", "eval")}
    return lab, rs, frames, sat, phi, lidx, rows


@torch.no_grad()
def predict(net, D, srows, alpha, cfg):
    _, _, frames, sat, _, lidx, _ = D
    net.eval()
    out = []
    for i in range(0, len(srows), 128):
        j = srows[i:i + 128]
        a = torch.full((len(j),), alpha, device=frames.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = net(T.make_batch(frames[lidx[j]], a, False, cfg),
                     T.make_batch(sat[j], torch.zeros_like(a), False, cfg)).float()
        out.append(M.predict_deg(lg))
    return torch.cat(out)


def evaluate(net, D, role, cfg, by_region=False) -> dict:
    _, rs, _, _, phi, lidx, rows = D
    srows = rows[role]
    errs = {a: M.angular_error_deg(predict(net, D, srows, a, cfg),
                                   torch.remainder(phi[lidx[srows]] - a, 360.0)).cpu().numpy()
            for a in T.EVAL_ANGLES}
    e0, allv = errs[0.0], np.concatenate(list(errs.values()))
    res = {"n": len(srows), "median_err_0": float(np.median(e0)), "acc10_0": float((e0 <= 10).mean()),
           "acc20_0": float((e0 <= 20).mean()), "acc45_0": float((e0 <= 45).mean()),
           "median_err_all": float(np.median(allv)), "acc10_all": float((allv <= 10).mean()),
           "acc45_all": float((allv <= 45).mean())}
    if by_region:
        reg = rs.region.to_numpy()[srows.cpu().numpy()]
        res["by_region"] = {r: {"median_err_0": float(np.median(e0[reg == r])),
                                "acc10_0": float((e0[reg == r] <= 10).mean())} for r in np.unique(reg)}
    return res


def train(name, lr, cfg, D, epochs, seed=0) -> dict:
    _, _, frames, sat, phi, lidx, rows = D
    dev = frames.device
    torch.manual_seed(seed)
    net = M.C8RelRot(widths=cfg.widths).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    tr = rows["train"]
    spe = math.ceil(len(tr) / BATCH)
    total, warm = epochs * spe, max(1, min(2 * spe, epochs * spe // 5))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    g = torch.Generator(device=dev).manual_seed(seed)
    hist, best, t0 = [], (float("inf"), 0, None), time.time()
    for ep in range(epochs):
        net.train()
        perm = tr[torch.randperm(len(tr), generator=g, device=dev)]
        tot = 0.0
        for i in range(spe):
            j = perm[i * BATCH:(i + 1) * BATCH]
            alpha = torch.rand(len(j), generator=g, device=dev) * 360.0
            target = M.soft_targets(torch.remainder(phi[lidx[j]] - alpha, 360.0), sigma_bins=1.0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = net(T.make_batch(frames[lidx[j]], alpha, True, cfg),
                         T.make_batch(sat[j], torch.zeros_like(alpha), True, cfg)).float()
            loss = -(target * F.log_softmax(lg, 1)).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        v = evaluate(net, D, "val", cfg)
        hist.append({"epoch": ep + 1, "loss": tot / spe, **{k: v[k] for k in ("median_err_all", "median_err_0",
                                                                             "acc10_0", "acc45_0")}})
        if v["median_err_all"] < best[0]:
            best = (v["median_err_all"], ep + 1, {k: t.detach().clone() for k, t in net.state_dict().items()})
        print(f"  deney {name} epoch {ep + 1}: kayıp {tot / spe:.3f}, val medyan hata (12 açı) "
              f"{v['median_err_all']:.2f}°, (0°) {v['median_err_0']:.2f}° [{time.time() - t0:.0f} s]", flush=True)
    net.load_state_dict(best[2])
    res = {"lr": lr, "cfg": asdict(cfg), "epochs": epochs, "best_epoch": best[1], "n_params": net.n_params(),
           "history": hist, "val": evaluate(net, D, "val", cfg, True), "calib": evaluate(net, D, "eval", cfg, True),
           "sec": round(time.time() - t0, 1)}
    T.MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "lr": lr, "cfg": asdict(cfg)}, T.MODELS / f"m1c_{name}.pt")
    return res


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    D = load("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    out_p = OUT / "m1c.json"
    res = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    for name, (lr, cfg) in CONFIGS.items():
        if a.only and name not in a.only:
            continue
        print(f"deney {name} başlıyor: lr {lr:g} {asdict(cfg)}", flush=True)
        r = train(name, lr, cfg, D, a.epochs)
        res[name] = r
        v, c = r["val"], r["calib"]
        print(f"deney {name} bitti: en iyi epoch {r['best_epoch']}, val {v['median_err_all']:.1f}° (04; ≤10° "
              f"{v['acc10_0']:.3f}, ≤45° {v['acc45_0']:.3f}), calib {c['median_err_0']:.1f}° (05: "
              f"{c['by_region']['05']['median_err_0']:.1f}°, 11: {c['by_region']['11']['median_err_0']:.1f}°), "
              f"calib ≤10° {c['acc10_0']:.3f}, ≤45° {c['acc45_0']:.3f}", flush=True)
        out_p.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
