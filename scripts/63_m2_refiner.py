"""Faz 3D – öğrenilmiş inceltici: v0 hacim kırpıntısı → düzeltilmiş log-olasılık + "boş" sınıfı.

  python scripts/63_m2_refiner.py                   # eğitim 01, 08, 09, 10; erken durdurma val 04; calib raporu
  python scripts/63_m2_refiner.py --seeds 1 2

Girdi (62_m2_crops.py): 2 × 13 θ × 48 × 48 (±48 m) — c ağırlıklı oy ve yoğunluk; ağda 3 kanal: log1p(oy),
log1p(yoğunluk), oy / tepe. Küresel v0 metrikleri (kırpıntı dışını da gören: ρ, H, Σc, …) boş başlığına gider.
Çıkış: hücre logitleri (13, 48, 48) + boş logiti; softmax tümü üzerinde. Konum kaybı θ üzerinden marjinal
(yön etiketi Phi1 gürültülü, gdt çoğu zaman ±6° dışında): pozitif ∧ GT kırpıntıda → −log Σ_k w_k p(x_k)
(çift doğrusal 4 hücre); negatif ∨ GT dışarıda → −log p(boş).
Skorlar: S25 = argmax çevresinde 25 m içindeki olasılık kütlesi ("doğru olma olasılığı"), Snull = 1 − p(boş).
Hangisinin ana skor olacağı val 04'te seçilir (AUROC), calib'e bakılmaz. Artırım: xy'de 90° dönme ve yansıma
(yansımada θ ekseni de ters); 90° dönmede göreli θ değişmez.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc import metrics as M
from guvenloc import stats as S
from guvenloc.metrics import log_run

DATA, OUT, HALF = Path("data/m2"), Path("results/faz3/m2"), 24
TRAIN_REG, VAL_REG = ("01", "08", "09", "10"), "04"
GLOBAL = ["log1p_sum_c", "rho", "entropy_bits", "spread_m", "frac_dropped", "log1p_support", "support_frac"]
N_GLOBAL = len(GLOBAL)
M1_DECISIONS = "results/faz2/m1/roma_row4_decisions_D2_olcek_rot90_s2_calib.csv"
TAU_4A = 709.0


def gfeat(df: pd.DataFrame) -> np.ndarray:
    g = pd.DataFrame({"log1p_sum_c": np.log1p(df.sum_c), "rho": df.rho, "entropy_bits": df.entropy_bits.fillna(25) / 20,
                      "spread_m": df.spread_m.fillna(50) / 50, "frac_dropped": df.frac_dropped,
                      "log1p_support": np.log1p(df.support.clip(lower=0)), "support_frac": df.support_frac})
    return g[GLOBAL].to_numpy(np.float32)


def block(cin, cout):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.GELU(),
                         nn.Conv3d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.GELU())


class Refiner(nn.Module):
    """Küçük 3B U-Net (θ ekseni korunur, xy 48 → 24 → 12) + boş başlığı."""

    def __init__(self, c=(16, 32, 64), n_global=N_GLOBAL):
        super().__init__()
        self.e1, self.e2, self.e3 = block(3, c[0]), block(c[0], c[1]), block(c[1], c[2])
        self.d2, self.d1 = block(c[2] + c[1], c[1]), block(c[1] + c[0], c[0])
        self.out = nn.Conv3d(c[0], 1, 1)
        self.null = nn.Sequential(nn.Linear(2 * c[2] + n_global, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x, g):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool3d(e1, (1, 2, 2)))
        e3 = self.e3(F.max_pool3d(e2, (1, 2, 2)))
        d2 = self.d2(torch.cat([F.interpolate(e3, scale_factor=(1, 2, 2)), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=(1, 2, 2)), e1], 1))
        cells = self.out(d1)[:, 0]                                            # (B, T, H, W)
        pooled = torch.cat([e3.mean((2, 3, 4)), e3.amax((2, 3, 4)), g], 1)
        return cells, self.null(pooled)[:, 0]


def prep(x16: torch.Tensor) -> torch.Tensor:
    x = x16.float()
    v, d = x[:, 0], x[:, 1]
    peak = v.flatten(1).amax(1).clamp(min=1e-6).view(-1, 1, 1, 1)
    return torch.stack([torch.log1p(v.clamp(min=0)), torch.log1p(d.clamp(min=0)), v / peak], 1)


def target_map(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(N, 48, 48) çift doğrusal GT ağırlıkları + (N,) boş hedefi (GT yok / dışarıda)."""
    n = 2 * HALF
    t = np.zeros((len(df), n, n), np.float32)
    gx, gy = df.gx.to_numpy(), df.gy.to_numpy()
    null = ~(np.isfinite(gx) & (gx >= 0) & (gx <= n - 1) & (gy >= 0) & (gy <= n - 1))
    for i in np.flatnonzero(~null):
        x0, y0 = int(np.floor(gx[i])), int(np.floor(gy[i]))
        fx, fy = gx[i] - x0, gy[i] - y0
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            if x0 + dx < n and y0 + dy < n:
                t[i, y0 + dy, x0 + dx] += w
    return t, null


def augment(x, tmap):
    k = int(torch.randint(0, 4, (1,)))
    x, tmap = torch.rot90(x, k, (-2, -1)), torch.rot90(tmap, k, (-2, -1))
    if torch.rand(1) < 0.5:
        x, tmap = torch.flip(x, (-1, -3)), torch.flip(tmap, (-1,))        # x yansıması + θ ekseni ters
    return x, tmap


def loss_fn(cells, nl, tmap, null):
    B = cells.shape[0]
    logits = torch.cat([cells.flatten(1), nl[:, None]], 1)
    logp = torch.log_softmax(logits.float(), 1)
    lxy = torch.logsumexp(logp[:, :-1].view_as(cells), 1).flatten(1)       # θ marjinali
    tw = tmap.flatten(1)
    pos = -torch.logsumexp(lxy + torch.log(tw.clamp(min=1e-12)), 1)
    neg = -logp[:, -1]
    return torch.where(null, neg, pos).mean() if B else cells.sum() * 0


@torch.no_grad()
def predict(net, X, G, dev, bs=256) -> dict:
    net.eval()
    out = {"ix": [], "iy": [], "s25": [], "snull": []}
    n = 2 * HALF
    yy, xx = torch.meshgrid(torch.arange(n, device=dev), torch.arange(n, device=dev), indexing="ij")
    for i in range(0, len(X), bs):
        x = prep(X[i:i + bs].to(dev, non_blocking=True))
        g = G[i:i + bs].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cells, nl = net(x, g)
        logp = torch.log_softmax(torch.cat([cells.float().flatten(1), nl.float()[:, None]], 1), 1)
        pxy = torch.logsumexp(logp[:, :-1].view(cells.shape), 1).exp()       # (B, H, W)
        k = pxy.flatten(1).argmax(1)
        iy, ix = k // n, k % n
        d2 = (xx[None] - ix[:, None, None]) ** 2 + (yy[None] - iy[:, None, None]) ** 2
        out["ix"].append(ix.cpu())
        out["iy"].append(iy.cpu())
        out["s25"].append((pxy * (d2 * 4.0 <= 25.0 ** 2)).flatten(1).sum(1).cpu())   # hücre 2 m
        out["snull"].append(1 - logp[:, -1].exp().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def positions(df, pr):
    """Kırpıntı hücresi → karo pikseli; pozitiflerde hata (m)."""
    x = ((df.cx.to_numpy() - HALF + pr["ix"]) + 0.5) * df.cell_px.to_numpy() - 0.5
    y = ((df.cy.to_numpy() - HALF + pr["iy"]) + 0.5) * df.cell_px.to_numpy() - 0.5
    return x, y


def load_split(split, pairs):
    df = pd.read_csv(DATA / f"crops_{split}.csv").join(
        pairs[["label", "region", "gt_x", "gt_y"]], on="pair_id")
    df["is_pos"] = df.label == "pos"
    X = torch.from_numpy(np.load(DATA / f"crops_{split}.npy", mmap_mode="r")[:])
    return df, X


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    dev = torch.device("cuda")
    pairs = pd.read_csv("data/pairs/pairs_v1.csv", dtype={"region": str}).set_index("pair_id")
    tr_df, tr_X = load_split("train", pairs)
    ca_df, ca_X = load_split("calib", pairs)
    tmap_all, null_all = target_map(tr_df)
    G_tr, G_ca = torch.from_numpy(gfeat(tr_df)), torch.from_numpy(gfeat(ca_df))
    itr = np.flatnonzero(tr_df.region.isin(TRAIN_REG).to_numpy())
    iva = np.flatnonzero((tr_df.region == VAL_REG).to_numpy())
    Xg = tr_X[itr].to(dev)                                                   # ≈ 3,9 GB float16
    Gg, Tg = G_tr[itr].to(dev), torch.from_numpy(tmap_all[itr]).to(dev)
    Ng = torch.from_numpy(null_all[itr]).to(dev)
    va = tr_df.iloc[iva]
    print(f"eğitim {len(itr)} (çift, dönme), val 04 {len(iva)}, calib {len(ca_df)}; "
          f"eğitimde GT kırpıntıda %{100 * (~null_all[itr]).mean():.1f}", flush=True)

    results = {}
    for seed in a.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = Refiner().to(dev)
        n_par = sum(p.numel() for p in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        steps = a.epochs * (len(itr) // a.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
        best, best_state, hist, t0 = -1.0, None, [], time.time()
        for ep in range(1, a.epochs + 1):
            net.train()
            perm = torch.randperm(len(itr), device=dev)
            tot, nb = 0.0, 0
            for b in range(len(itr) // a.batch):
                j = perm[b * a.batch:(b + 1) * a.batch]
                x, tm = augment(prep(Xg[j]), Tg[j])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    cells, nl = net(x, Gg[j])
                loss = loss_fn(cells, nl, tm, Ng[j])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                tot, nb = tot + float(loss.detach()), nb + 1
            pr = predict(net, tr_X[iva], G_tr[iva], dev)
            x, y = positions(va, pr)
            err = np.where(va.is_pos, np.hypot(x - va.gt_x, y - va.gt_y) * va.mpp, np.nan)
            ok = M.good_label(va.is_pos, err)
            auc25, aucn = M.auroc(pr["s25"], ok), M.auroc(pr["snull"], ok)
            ceil = float(ok[va.is_pos.to_numpy()].mean())
            hist.append({"epoch": ep, "loss": tot / nb, "val_auroc_s25": auc25, "val_auroc_snull": aucn, "val_ceiling": ceil})
            print(f"tohum {seed} epoch {ep}/{a.epochs}: kayıp {tot / nb:.4f}, val AUROC S25 {auc25:.4f}, Snull {aucn:.4f}, "
                  f"val başarı tavanı {ceil:.3f} [{time.time() - t0:.0f} s]", flush=True)
            score = max(auc25, aucn)
            if score > best:
                best, best_ep = score, ep
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        Path("cache/models").mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "epoch": best_ep, "hist": hist}, f"cache/models/m2_refiner_s{seed}.pt")
        hb = hist[best_ep - 1]
        main_score = "s25" if hb["val_auroc_s25"] >= hb["val_auroc_snull"] else "snull"
        results[seed] = calib_report(net, ca_df, ca_X, G_ca, dev, main_score, seed) | {
            "best_epoch": best_ep, "val": hb, "main_score": main_score, "n_params": n_par, "hist": hist}
        print(f"tohum {seed} bitti: en iyi epoch {best_ep}, ana skor {main_score} (val), "
              f"calib: {json.dumps(results[seed]['summary'], ensure_ascii=False)}", flush=True)
    (OUT / "refiner_calib.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float),
                                             encoding="utf-8")
    write_md(results)
    print("M2R_BITTI", flush=True)
    return 0


def calib_report(net, ca_df, ca_X, G_ca, dev, main_score, seed) -> dict:
    """Calib: 4a dönmesi (M1-C v0.1) üzerinde 4a (RANSAC+inlier), 5a (v0 hacim + destek) ve 5r (inceltici)."""
    pr = predict(net, ca_X, G_ca, dev)
    x, y = positions(ca_df, pr)
    d = ca_df.assign(rx=x, ry=y, S=pr[main_score], s25=pr["s25"], snull=pr["snull"])
    d["err_r"] = np.where(d.is_pos, np.hypot(d.rx - d.gt_x, d.ry - d.gt_y) * d.mpp, np.nan)
    rf = pd.read_csv("results/faz1/baselines/feats_v1_calib_px5.csv")[["pair_id", "rot", "n_inliers", "cx", "cy", "valid"]]
    d = d.merge(rf.rename(columns={"cx": "rcx", "cy": "rcy"}), on=["pair_id", "rot"])
    e_ran = np.where(d.valid & (d.n_inliers > 0), np.hypot(d.rcx - d.gt_x, d.rcy - d.gt_y) * d.mpp, np.inf)
    d["err_ransac"] = np.where(d.is_pos, e_ran, np.nan)
    dec = pd.read_csv(M1_DECISIONS)
    rot4a = dec[dec.row == "4a"].set_index("pair_id").rot
    a4 = d.set_index(["pair_id", "rot"]).loc[list(zip(rot4a.index, rot4a.to_numpy(), strict=True))].reset_index()
    a4["block"] = a4.region + "_" + (a4.pair_id.str.split("_").str[1].astype(int) // 50).astype(str)
    ok4, ok0, okr = (M.good_label(a4.is_pos, a4[c]) for c in ("err_ransac", "err_m", "err_r"))
    isp = a4.is_pos.to_numpy()
    rows = {"4a": (a4.n_inliers.to_numpy(float), ok4), "5a_v0": (a4.support.to_numpy(float), ok0),
            "5r": (a4.S.to_numpy(float), okr)}
    summ = {}
    for k, (sc, ok) in rows.items():
        summ[k] = {"auroc": M.auroc(sc, ok), "ceiling@25": float(ok[isp].mean()),
                   "auroc_same_label_ransac": M.auroc(sc, ok4)}
    bl = a4.block.to_numpy()
    sc_r, sc_4 = rows["5r"][0], rows["4a"][0]
    dauc = S.cluster_bootstrap_ci(lambda i: M.auroc(sc_r[i], okr[i]) - M.auroc(sc_4[i], ok4[i]), bl, B=2000, seed=seed)
    dsame = S.cluster_bootstrap_ci(lambda i: M.auroc(sc_r[i], ok4[i]) - M.auroc(sc_4[i], ok4[i]), bl, B=2000, seed=seed)
    # yanlış kabul, eşit kabul oranında (4a'nın τ = 709 kabul oranı): tavan farkından bağımsız karşılaştırma
    acc4 = sc_4 >= TAU_4A
    q = 1 - acc4.mean()
    accr = sc_r >= np.quantile(sc_r, q)
    fa4, far = acc4 & ~ok4, accr & ~okr
    mc = S.mcnemar(~far, ~fa4)
    summ["equal_accept_rate"] = {"accept_rate": float(acc4.mean()), "fa_4a": float(fa4.mean()), "fa_5r": float(far.mean()),
                                 "succ_4a": float((acc4 & ok4)[isp].mean()), "succ_5r": float((accr & okr)[isp].mean()),
                                 "mcnemar": mc}
    summ["auroc_diff_5r_4a"] = list(dauc)
    summ["auroc_diff_same_label"] = list(dsame)
    log_run("63_m2_refiner.py", Path("configs/m2.yaml"), seed, "calib", "row5r_auroc", round(summ["5r"]["auroc"], 5))
    return {"summary": summ}


def write_md(results):
    lines = ["# Faz 3D — öğrenilmiş inceltici, calib (4a dönmesi, M1-C v0.1)", "",
             "| tohum | ana skor (val) | val AUROC | satır | AUROC (kendi etiketi) | AUROC (RANSAC etiketi) | başarı tavanı |",
             "|---|---|---|---|---|---|---|"]
    for seed, r in results.items():
        s = r["summary"]
        for k in ("4a", "5a_v0", "5r"):
            lines.append(f"| {seed} | {r['main_score']} | {max(r['val']['val_auroc_s25'], r['val']['val_auroc_snull']):.4f} | {k} | "
                         f"{s[k]['auroc']:.4f} | {s[k]['auroc_same_label_ransac']:.4f} | {s[k]['ceiling@25']:.3f} |")
    lines.append("")
    for seed, r in results.items():
        s = r["summary"]
        e = s["equal_accept_rate"]
        lines += [(f"Tohum {seed}: AUROC farkı 5r − 4a {s['auroc_diff_5r_4a'][0]:+.4f} [{s['auroc_diff_5r_4a'][1]:+.4f}, "
                  f"{s['auroc_diff_5r_4a'][2]:+.4f}]; aynı etiketle {s['auroc_diff_same_label'][0]:+.4f} "
                  f"[{s['auroc_diff_same_label'][1]:+.4f}, {s['auroc_diff_same_label'][2]:+.4f}]. Eşit kabul oranında "
                  f"(%{100 * e['accept_rate']:.1f}): yanlış kabul 4a {e['fa_4a']:.3f} / 5r {e['fa_5r']:.3f}, başarı@25 "
                  f"{e['succ_4a']:.3f} / {e['succ_5r']:.3f}, McNemar p = {e['mcnemar']['p']:.2e} "
                  f"(yalnız 4a yanlış kabul {e['mcnemar']['a_only']}, yalnız 5r {e['mcnemar']['b_only']})."), ""]
    (OUT / "refiner_calib.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
