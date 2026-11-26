"""Faz 2C (öneri C fizibilitesi) – göreli dönme modeli (M.C8RelRot): drone karesi + kuzey-yukarı uydu karosu → φ.

  python scripts/55_m1c_train.py                     # 3 yapılandırma, tohum 0 → results/faz2/m1/m1c.json
  python scripts/55_m1c_train.py --only C1_1e-3
  python scripts/55_m1c_train.py --veri gercek       # sistemdeki gerçek karo (513 px) → m1c_gercek.json

--veri gercek: karo = pairs_v1 pozitif karosu (ayak izi köşegeni + 400 m ≈ 3× merkez karesi, kayma ≤ 200 m),
54_m1c_prep.py --gercek. Uydu karosu döndürülmez, maskelenmez, yeniden boyutlandırılmaz (513 px); renk
oynatması drone ile aynı, ölçek oynatması yok. C4 = küresel ortalama (C2 ile aynı model), C5 = kayan pencere.

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
CONFIGS_GERCEK = {
    "C4_ortalama": (3e-3, T.Cfg("C4", aug="strong", early_stop=True), "mean"),
    "C5_pencere": (3e-3, T.Cfg("C5", aug="strong", early_stop=True), "window"),
}
CONFIGS = {
    "C1_1e-3": (1e-3, T.Cfg("C1", aug="strong", early_stop=True)),
    "C2_3e-3": (3e-3, T.Cfg("C2", aug="strong", early_stop=True)),
    "C3_1e-3_gri": (1e-3, T.Cfg("C3", aug="strong", gray=True, early_stop=True)),
}


SAT = {"kolay": "data/m1/relsat_321", "gercek": "data/m1/relsat_real_513"}
MODE = {"veri": "kolay"}


def make_sat(imgs_uint8, train: bool, cfg):
    """Gerçek karo: renk oynatması + normalize; döndürme/maske/boyutlandırma yok."""
    if MODE["veri"] == "kolay":
        return T.make_batch(imgs_uint8, torch.zeros(len(imgs_uint8), device=imgs_uint8.device), train, cfg)
    x = imgs_uint8.permute(0, 3, 1, 2).float() / 255.0
    B, dev = len(x), x.device
    if train:
        def u(lo, hi, *shape):
            return lo + (hi - lo) * torch.rand(B, *shape, device=dev)
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - m) * u(0.6, 1.4, 1, 1, 1) + m) * u(0.6, 1.4, 1, 1, 1)
        g = x.mean(1, keepdim=True)
        x = (g + (x - g) * u(0.5, 1.5, 1, 1, 1)) * u(0.9, 1.1, 3, 1, 1)
        x = x.clamp(0, 1)
    if cfg.gray:
        x = (0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).expand(-1, 3, -1, -1)
    mean = torch.tensor(T.MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(T.STD, device=dev).view(1, 3, 1, 1)
    return (x - mean) / std


def load(dev):
    lab = pd.read_csv("data/m1/labels_v2.csv", dtype={"region": str})
    rs = pd.read_csv(SAT[MODE["veri"]] + ".csv", dtype={"region": str})
    frames = torch.from_numpy(np.load("data/m1/frames_321.npy")).to(dev)
    sat = torch.from_numpy(np.load(SAT[MODE["veri"]] + ".npy")).to(dev)
    phi = torch.tensor(lab.phi_corr.to_numpy(), dtype=torch.float32, device=dev)
    lidx = torch.tensor(rs.lab_idx.to_numpy(), device=dev)
    rows = {r: torch.tensor(np.flatnonzero(rs.m1_role.to_numpy() == r), device=dev) for r in ("train", "val", "eval")}
    return lab, rs, frames, sat, phi, lidx, rows


@torch.no_grad()
def predict_all(net, D, srows, cfg) -> dict:
    """Her açı için tahmin; uydu betimleyicisi parça başına bir kez hesaplanır."""
    _, _, frames, sat, _, lidx, _ = D
    net.eval()
    out = {a: [] for a in T.EVAL_ANGLES}
    for i in range(0, len(srows), 64):
        j = srows[i:i + 64]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ds = net.describe_sat(make_sat(sat[j], False, cfg))
            for a in T.EVAL_ANGLES:
                al = torch.full((len(j),), a, device=frames.device)
                lg = net.score(net.describe(T.make_batch(frames[lidx[j]], al, False, cfg)), ds).float()
                out[a].append(M.predict_deg(lg))
    return {a: torch.cat(v) for a, v in out.items()}


def evaluate(net, D, role, cfg, by_region=False) -> dict:
    _, rs, _, _, phi, lidx, rows = D
    srows = rows[role]
    preds = predict_all(net, D, srows, cfg)
    errs = {a: M.angular_error_deg(p, torch.remainder(phi[lidx[srows]] - a, 360.0)).cpu().numpy()
            for a, p in preds.items()}
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


def train(name, lr, cfg, D, epochs, seed=0, sat_pool="mean") -> dict:
    _, _, frames, sat, phi, lidx, rows = D
    dev = frames.device
    torch.manual_seed(seed)
    net = M.C8RelRot(widths=cfg.widths, sat_pool=sat_pool).to(dev)
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
                         make_sat(sat[j], True, cfg)).float()
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
    res = {"lr": lr, "cfg": asdict(cfg), "sat_pool": sat_pool, "veri": MODE["veri"], "epochs": epochs, "best_epoch": best[1], "n_params": net.n_params(),
           "history": hist, "val": evaluate(net, D, "val", cfg, True), "calib": evaluate(net, D, "eval", cfg, True),
           "sec": round(time.time() - t0, 1)}
    T.MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "lr": lr, "cfg": asdict(cfg), "sat_pool": sat_pool}, T.MODELS / f"m1c_{name}.pt")
    return res


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--veri", choices=["kolay", "gercek"], default="kolay")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    MODE["veri"] = a.veri
    global BATCH
    BATCH = a.batch
    torch.backends.cudnn.benchmark = True
    D = load("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    out_p = OUT / ("m1c.json" if a.veri == "kolay" else "m1c_gercek.json")
    configs = ({k: (lr, c, "mean") for k, (lr, c) in CONFIGS.items()} if a.veri == "kolay" else CONFIGS_GERCEK)
    res = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    for name, (lr, cfg, pool) in configs.items():
        if a.only and name not in a.only:
            continue
        print(f"deney {name} başlıyor: lr {lr:g} {asdict(cfg)}", flush=True)
        r = train(name, lr, cfg, D, a.epochs, sat_pool=pool)
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
