"""Faz 2C – Modül 1 eğitimi, öğrenme oranı seçimi (doğrulama 04) ve calib değerlendirmesi.

  python scripts/52_m1_train.py all                 # lr taraması (tohum 0) → seçim → tohum 1, 2 → calib
  python scripts/52_m1_train.py run --lr 1e-3 --seed 0
  python scripts/52_m1_train.py ab                  # öneri A/B kısa deneyleri (tohum 0) → m1_ab.json

ab modu (docs/faz2_m1.md 2C-v2): A = daha sert düzenleme (güçlü renk/ölçek/bulanıklık oynatma, gri ton,
küçük ağ, doğrulamada erken durdurma); B = eğitime kuzey-yukarı uydu kırpıntıları (φ = 0,
scripts/53_m1_sat_prep.py; yalnız eğitim bölgeleri). Yapılandırma seçimi yalnız val 04'e göre; calib her
yapılandırma için TEŞHİS amaçlı raporlanır.

Veri: data/m1/frames_321.npy + labels_v2.csv. Eğitim yalnız m1_role == train (01, 08, 09, 10); seçim
m1_role == val (04); değerlendirme m1_role == eval (calib 05, 11; etiketler ham Phi1). Test bölgeleri yok.
Hedef etiket φ_corr (karar F2-4). Her örnekte içerik U(0, 360) saat yönünde döndürülür → φ' = φ − α.
Kayıp: dairesel Gauss yumuşatmalı çapraz entropi (σ = 1 bölme). λ·L_prior kullanılmaz: denetimli kurulumda
önselin hedefi (kuzey-yukarı) etiketin kendisi, ayrı bir şey ayarlamıyor (docs/faz2_m1.md 2C).
Değerlendirme açıları: 0° (kare olduğu gibi — gerçek kullanım) ve 12 sabit açı (k·30°).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc import m1_canon as M
from guvenloc.metrics import log_run

OUT = Path("results/faz2/m1")
MODELS = Path("cache/models")
LR_GRID = [3e-4, 1e-3, 3e-3]
SEEDS = [0, 1, 2]
EPOCHS = 60
BATCH = 64
EVAL_ANGLES = [30.0 * k for k in range(12)]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Cfg:
    """Eğitim yapılandırması. Varsayılanlar = v1 (docs/faz2_m1.md 2C)."""
    name: str = "v1"
    widths: tuple[int, ...] = (16, 32, 64, 96, 128)
    aug: str = "light"          # light: parlaklık/kontrast ±0,2 | strong: + doygunluk, kanal kazancı, ölçek, bulanıklık
    gray: bool = False          # girdiyi gri tona çevir (renk/mevsim ipucunu kapat)
    sat_frac: float = 0.0       # yığının bu oranı uydu kırpıntısı (φ = 0)
    early_stop: bool = False    # en iyi val (12 açı medyanı) durumunu geri yükle
    eval_every: int = 10
    weight_decay: float = 1e-4


AB_CONFIGS = [
    Cfg("v1_es", early_stop=True, eval_every=5),
    Cfg("A1_sert", aug="strong", early_stop=True, eval_every=5),
    Cfg("A2_sert_gri", aug="strong", gray=True, early_stop=True, eval_every=5),
    Cfg("A3_sert_gri_kucuk", widths=(8, 16, 32, 48, 64), aug="strong", gray=True, early_stop=True,
        eval_every=5, weight_decay=1e-3),
    Cfg("B1_sert_uydu50", aug="strong", sat_frac=0.5, early_stop=True, eval_every=5),
    Cfg("B2_sert_gri_uydu50", aug="strong", gray=True, sat_frac=0.5, early_stop=True, eval_every=5),
    Cfg("B3_yalniz_uydu", aug="strong", sat_frac=1.0, early_stop=True, eval_every=5),
]


def load_data(device):
    lab = pd.read_csv("data/m1/labels_v2.csv", dtype={"region": str})
    frames = torch.from_numpy(np.load("data/m1/frames_321.npy")).to(device)   # (N,321,321,3) uint8
    phi = torch.tensor(lab.phi_corr.to_numpy(), dtype=torch.float32, device=device)
    roles = {r: torch.tensor(np.flatnonzero(lab.m1_role.to_numpy() == r), device=device)
             for r in ("train", "val", "eval")}
    sat_p = Path("data/m1/sat_321.npy")
    sat = torch.from_numpy(np.load(sat_p)).to(device) if sat_p.exists() else None
    return lab, frames, phi, roles, sat


def _gauss_blur(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    k = torch.arange(-2, 3, device=x.device, dtype=x.dtype)
    w = torch.exp(-0.5 * (k / sigma) ** 2)
    w = w / w.sum()
    C = x.shape[1]
    x = F.conv2d(x, w.view(1, 1, 1, 5).repeat(C, 1, 1, 1), padding=(0, 2), groups=C)
    return F.conv2d(x, w.view(1, 1, 5, 1).repeat(C, 1, 1, 1), padding=(2, 0), groups=C)


def make_batch(imgs_uint8, alpha_cw, train: bool, cfg: Cfg = Cfg()):
    """(B,H,W,3) uint8 → döndürülmüş, maskeli, normalize (B,3,257,257)."""
    x = imgs_uint8.permute(0, 3, 1, 2).float() / 255.0
    B, dev = len(x), x.device
    zoom = None

    def u(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(B, *shape, device=dev)

    if train:
        amp = 0.2 if cfg.aug == "light" else 0.4
        m = x.mean(dim=(1, 2, 3), keepdim=True)
        x = ((x - m) * u(1 - amp, 1 + amp, 1, 1, 1) + m) * u(1 - amp, 1 + amp, 1, 1, 1)
        if cfg.aug == "strong":
            g = x.mean(1, keepdim=True)
            x = g + (x - g) * u(0.5, 1.5, 1, 1, 1)               # doygunluk
            x = x * u(0.9, 1.1, 3, 1, 1)                         # kanal kazancı (renk sıcaklığı)
            blur = (torch.rand(B, device=dev) < 0.3).view(B, 1, 1, 1)
            x = torch.where(blur, _gauss_blur(x).float(), x)   # autocast altında conv bf16 döner
            zoom = torch.exp(u(math.log(0.8), math.log(1.25)))
        x = x.clamp(0, 1)
    if cfg.gray:
        x = (0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).expand(-1, 3, -1, -1)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return M.rotate_cw((x - mean) / std, alpha_cw, out_size=M.IN_SIZE, zoom=zoom)


@torch.no_grad()
def predict(net, frames, idx, alpha_cw: float, cfg: Cfg = Cfg()) -> torch.Tensor:
    net.eval()
    out = []
    for i in range(0, len(idx), 128):
        j = idx[i:i + 128]
        a = torch.full((len(j),), alpha_cw, device=frames.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(make_batch(frames[j], a, train=False, cfg=cfg)).float()
        out.append(M.predict_deg(logits))
    return torch.cat(out)


def evaluate(net, frames, phi, idx, lab=None, cfg: Cfg = Cfg()) -> dict:
    errs = {}
    preds = {}
    for a in EVAL_ANGLES:
        p = predict(net, frames, idx, a, cfg)
        preds[a] = p
        errs[a] = M.angular_error_deg(p, torch.remainder(phi[idx] - a, 360.0)).cpu().numpy()
    allv = np.concatenate(list(errs.values()))
    e0 = errs[0.0]
    res = {"n": len(idx), "median_err_0": float(np.median(e0)), "mean_err_0": float(np.mean(e0)),
           "acc10_0": float((e0 <= 10).mean()), "acc20_0": float((e0 <= 20).mean()),
           "acc45_0": float((e0 <= 45).mean()), "median_err_all": float(np.median(allv)),
           "acc10_all": float((allv <= 10).mean()),
           "median_err_by_angle": {f"{a:g}": float(np.median(v)) for a, v in errs.items()}}
    # eşdeğerlik tutarlılığı (etiketsiz): θ̂(R_α I) + α ile θ̂(I) farkı
    eq = {}
    for a in (45.0, 20.0, 90.0):
        p = predict(net, frames, idx, a, cfg)
        eq[f"{a:g}"] = float(np.median(M.angular_error_deg(p + a, preds[0.0]).cpu().numpy()))
    res["equiv_median_dev"] = eq
    if lab is not None:
        reg = lab.region.to_numpy()[idx.cpu().numpy()]
        res["by_region_median_err_0"] = {r: float(np.median(e0[reg == r])) for r in np.unique(reg)}
    return res


def train_run(lr: float, seed: int, data, epochs: int = EPOCHS, save: bool = True, cfg: Cfg = Cfg(),
              tag: str | None = None) -> dict:
    lab, frames, phi, roles, sat = data
    if cfg.sat_frac > 0 and sat is None:
        raise FileNotFoundError("data/m1/sat_321.npy yok → önce scripts/53_m1_sat_prep.py")
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = frames.device
    net = M.C8Canon(widths=cfg.widths).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    tr = roles["train"]
    steps_per_epoch = math.ceil(len(tr) / BATCH)
    n_sat = round(cfg.sat_frac * BATCH)
    total = epochs * steps_per_epoch
    warm = max(1, min(2 * steps_per_epoch, total // 5))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    g = torch.Generator(device=dev).manual_seed(seed)
    hist = []
    best = (float("inf"), 0, None)
    label = tag or f"lr {lr:g} tohum {seed}"
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = tr[torch.randperm(len(tr), generator=g, device=dev)]
        tot = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * (BATCH - n_sat):(i + 1) * (BATCH - n_sat)] if n_sat < BATCH else perm[:0]
            imgs, phis = frames[idx], phi[idx]
            if n_sat:
                js = torch.randint(len(sat), (n_sat,), generator=g, device=dev)
                imgs = torch.cat([imgs, sat[js]])
                phis = torch.cat([phis, torch.zeros(n_sat, device=dev)])
            alpha = torch.rand(len(imgs), generator=g, device=dev) * 360.0
            target = M.soft_targets(torch.remainder(phis - alpha, 360.0), sigma_bins=1.0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(make_batch(imgs, alpha, train=True, cfg=cfg)).float()
            loss = -(target * F.log_softmax(logits, 1)).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        if (ep + 1) % cfg.eval_every == 0 or ep == epochs - 1:
            v = evaluate(net, frames, phi, roles["val"], cfg=cfg)
            hist.append({"epoch": ep + 1, "loss": tot / steps_per_epoch, "val_median_err_all": v["median_err_all"],
                         "val_median_err_0": v["median_err_0"]})
            if cfg.early_stop and v["median_err_all"] < best[0]:
                best = (v["median_err_all"], ep + 1,
                        {k: t.detach().clone() for k, t in net.state_dict().items()})
            print(f"  {label} epoch {ep + 1}: kayıp {tot / steps_per_epoch:.3f}, "
                  f"val medyan hata (12 açı) {v['median_err_all']:.2f}°, (0°) {v['median_err_0']:.2f}° "
                  f"[{time.time() - t0:.0f} s]", flush=True)
    if cfg.early_stop and best[2] is not None:
        net.load_state_dict(best[2])
    res = {"lr": lr, "seed": seed, "epochs": epochs, "cfg": asdict(cfg), "n_params": net.n_params(),
           "history": hist, "best_epoch": best[1] if cfg.early_stop else epochs,
           "val": evaluate(net, frames, phi, roles["val"], lab, cfg=cfg), "sec": round(time.time() - t0, 1)}
    if save:
        MODELS.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": net.state_dict(), "lr": lr, "seed": seed, "epochs": epochs,
                    "widths": cfg.widths, "cfg": asdict(cfg)}, MODELS / f"m1_{cfg.name}_lr{lr:g}_seed{seed}.pt")
    res["_net"] = net
    return res


def run_ab(data, lr: float, epochs: int, only: list[str] | None) -> dict:
    """Öneri A/B kısa deneyleri: tohum 0, sabit lr; seçim val 04'te, calib teşhis."""
    lab, frames, phi, roles, _ = data
    out_p = OUT / "m1_ab.json"
    res = json.loads(out_p.read_text(encoding="utf-8")) if out_p.exists() else {}
    for cfg in AB_CONFIGS:
        if only and cfg.name not in only:
            continue
        print(f"deney {cfg.name} başlıyor: {asdict(cfg)}", flush=True)
        r = train_run(lr, 0, data, epochs, cfg=cfg, tag=f"deney {cfg.name}")
        net = r.pop("_net")
        r["calib"] = evaluate(net, frames, phi, roles["eval"], lab, cfg=cfg)
        res[cfg.name] = r
        print(f"deney {cfg.name} bitti: en iyi epoch {r['best_epoch']}, val {r['val']['median_err_all']:.1f}° "
              f"(04), calib {r['calib']['median_err_0']:.1f}° (05: "
              f"{r['calib']['by_region_median_err_0']['05']:.1f}°, 11: "
              f"{r['calib']['by_region_median_err_0']['11']:.1f}°), calib ≤10° oranı "
              f"{r['calib']['acc10_0']:.3f}", flush=True)
        out_p.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        del net
        torch.cuda.empty_cache()
    best = min(res, key=lambda k: res[k]["val"]["median_err_all"])
    print(f"val'e göre en iyi yapılandırma: {best} ({res[best]['val']['median_err_all']:.1f}°)", flush=True)
    return res


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "run", "ab"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    ap.add_argument("--only", nargs="*", help="ab: yalnız bu yapılandırmalar")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    torch.backends.cudnn.benchmark = True
    data = load_data("cuda")
    OUT.mkdir(parents=True, exist_ok=True)

    if a.cmd == "ab":
        run_ab(data, a.lr, a.epochs, a.only)
        return 0

    if a.cmd == "run":
        r = train_run(a.lr, a.seed, data, a.epochs)
        r.pop("_net")
        print(json.dumps(r["val"], indent=1))
        return 0

    # 1) lr taraması (tohum 0), seçim doğrulamada (04), 12 açı medyan hata
    sweep = {}
    nets = {}
    for lr in LR_GRID:
        r = train_run(lr, 0, data, a.epochs)
        nets[lr] = r.pop("_net")
        sweep[lr] = r
    best = min(LR_GRID, key=lambda lr: sweep[lr]["val"]["median_err_all"])
    print(f"seçilen lr = {best:g} (val medyan hata, 12 açı: "
          + ", ".join(f"{lr:g}: {sweep[lr]['val']['median_err_all']:.2f}°" for lr in LR_GRID) + ")")
    # 2) seçilen lr ile tohum 1, 2
    runs = {0: (sweep[best], nets[best])}
    for s in SEEDS[1:]:
        r = train_run(best, s, data, a.epochs)
        runs[s] = (r, r.pop("_net"))
    # 3) calib değerlendirmesi
    lab, frames, phi, roles, _ = data
    ev = {s: evaluate(net, frames, phi, roles["eval"], lab) for s, (_, net) in runs.items()}
    keys = ["median_err_0", "acc10_0", "acc20_0", "acc45_0", "median_err_all", "acc10_all"]
    summ = {k: [float(np.mean([ev[s][k] for s in SEEDS])), float(np.std([ev[s][k] for s in SEEDS]))] for k in keys}
    res = {"lr_grid": LR_GRID, "sweep_val": {f"{lr:g}": sweep[lr]["val"] for lr in LR_GRID},
           "sweep_history": {f"{lr:g}": sweep[lr]["history"] for lr in LR_GRID}, "lr": best,
           "val_by_seed": {s: runs[s][0]["val"] for s in SEEDS}, "eval_by_seed": ev, "eval_summary": summ,
           "epochs": a.epochs, "batch": BATCH, "n_params": sweep[best]["n_params"]}
    (OUT / "m1_train.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    for s, (_, net) in runs.items():
        torch.save({"state_dict": net.state_dict(), "lr": best, "seed": s, "epochs": a.epochs,
                    "widths": (16, 32, 64, 96, 128)}, MODELS / f"m1_seed{s}.pt")
    print("calib (5 + 11), 3 tohum ortalama ± std:")
    for k, (m, sd) in summ.items():
        print(f"  {k}: {m:.3f} ± {sd:.3f}")
    seed = int(cfg["seed"])
    for k in ("median_err_0", "acc10_0", "median_err_all"):
        log_run("52_m1_train.py", a.config, seed, "calib", f"m1_{k}", round(summ[k][0], 4),
                notes=f"±{summ[k][1]:.4f} seeds={SEEDS} lr={best:g} epochs={a.epochs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
