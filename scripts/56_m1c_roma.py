"""Faz 2D – M1-C (göreli dönme, gerçek karo) → RoMa bağlama: satır 4 politikaları, RoMa önbelleğinden simülasyon.

  python scripts/56_m1c_roma.py                  # calib; eşikler eğitim negatiflerinden
  python scripts/56_m1c_roma.py --model C4_ortalama

Yeni RoMa çağrısı yok: her (çift, dönme) için RoMa özellikleri Faz 1F'de çıkarıldı
(results/faz1/baselines/feats_v1_{split}_px5.csv; dönme = drone'a saat yönünde r°). M1-C her calib çifti
(pozitif + zor/kolay negatif) için drone karesi + o çiftin karosu (pairs_v1, 513 px) ile 36 bölmeli dağılım
verir; dönme r'nin olasılığı = φ ∈ [r − 45°, r + 45°) kütlesi (drone saat yönünde φ döndürülünce kuzey-yukarı).

Satır 4 politikaları (çağrı/kare):
  4a  M1 tek çağrı: en olası dönme, karar inlier ≥ τ_4a                                  (1)
  4b  M1 ilk 2: en olası 2 dönme, en çok inlier'lı olan, karar inlier ≥ τ_4b              (2)
  4c  M1 ardışık: dönmeler M1 olasılık sırasıyla; ilk inlier ≥ τ_2 olan kabul, durulur;
      hiçbiri geçmezse ret (4 çağrı). τ_2 = satır 2 eşiği → negatiflerde kabul ⇔ max ≥ τ_2,
      FAR-neg satır 2 ile aynı; kazanç yalnız çağrı sayısında.                          (1–4)
  4d  M1 ardışık, en çok 2: en olası dönme inlier ≥ τ_4b ise kabul; değilse 2. dönme, ≥ τ_4b ise
      kabul, değilse ret. Kabul kararı 4b ile aynı (max ≥ τ_4b), çağrı daha az.          (1–2)
Eşikler (4a, 4b) eğitim negatiflerinde FAR-neg ≤ α ile; M1 negatiflerde de çalıştırılmalı ama eğitim
bölgelerinde örneklem içi → eğitimde dönme seçimi yerine: 4a için 4 dönmenin hepsi havuzlanır, 4b için
komşu dönme çiftlerinin maksimumu (M1'in ilk 2'si neredeyse hep komşu). Calib'e dokunulmaz.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from guvenloc import m1_canon as M
from guvenloc import stats as S
from guvenloc.io_visloc import crop_square_m, load_region
from guvenloc.metrics import log_run


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


B40 = _load("baselines40", "scripts/40_baselines.py")
C55 = _load("m1c55", "scripts/55_m1c_train.py")

OUT = Path("results/faz2/m1")
ROTS = (0, 90, 180, 270)
DATA = Path("data/UAV_VisLoc_dataset")


def tiles_for(pairs: pd.DataFrame, path: Path) -> np.ndarray:
    if path.exists():
        return np.load(path)
    regions = {r: load_region(DATA, r) for r in sorted(pairs.tile_region.unique())}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:
        arr = np.stack(list(ex.map(lambda r: crop_square_m(regions[r.tile_region], r.tile_lat, r.tile_lon,
                                                          r.tile_size_m, 513)[..., :3], pairs.itertuples())))
    np.save(path, arr)
    return arr


@torch.no_grad()
def m1_rot_probs(model: str, pairs: pd.DataFrame, tiles: np.ndarray) -> np.ndarray:
    """(N, 4) dönme olasılıkları (ROTS sırasıyla) + (N,) φ̂."""
    import re
    base = re.sub(r"_s\d+$", "", model)          # tohum eki (55_m1c_train.py --seeds)
    _lr, cfg, pool = (C55.CONFIGS_GERCEK[base] if base in C55.CONFIGS_GERCEK else C55.CONFIGS_V2[base][:3])
    C55.MODE["veri"] = "gercek"
    net = M.C8RelRot(sat_pool=pool).cuda()
    net.load_state_dict(torch.load(f"cache/models/m1c_{model}.pt", map_location="cuda")["state_dict"])
    net.eval()
    lab = pd.read_csv("data/m1/labels_v2.csv", dtype={"region": str})
    frames = np.load("data/m1/frames_321.npy", mmap_mode="r")
    fidx = pd.Series(np.arange(len(lab)), index=lab.drone_path)[pairs.drone_path].to_numpy()
    centers = torch.tensor(M.bin_centers_deg(), device="cuda", dtype=torch.float32)
    probs, phis = [], []
    for i in range(0, len(pairs), 64):
        d = torch.from_numpy(np.ascontiguousarray(frames[fidx[i:i + 64]])).cuda()
        s = torch.from_numpy(tiles[i:i + 64]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = net(C55.T.make_batch(d, torch.zeros(len(d), device="cuda"), False, cfg),
                     C55.make_sat(s, False, cfg)).float()
        p = lg.softmax(1)
        pr = torch.stack([(p * ((torch.remainder(centers - r + 45, 360) < 90).float())).sum(1) for r in ROTS], 1)
        probs.append(pr.cpu().numpy())
        phis.append(M.predict_deg(lg).cpu().numpy())
    return np.concatenate(probs), np.concatenate(phis)


def train_taus(alpha: float, tau2: int) -> dict:
    f = pd.read_csv("results/faz1/baselines/feats_v1_train_px5.csv")
    pairs = pd.read_csv("data/pairs/pairs_v1.csv", usecols=["pair_id", "label"])
    f = f.merge(pairs, on="pair_id")
    neg = f[f.label == "neg"].pivot(index="pair_id", columns="rot", values="n_inliers").dropna()
    single = neg.to_numpy().ravel()
    adj = np.concatenate([np.maximum(neg[r], neg[(r + 90) % 360]).to_numpy() for r in ROTS])
    return {"4a": B40.tau_for_far(single, alpha), "4b": B40.tau_for_far(adj, alpha), "4c": tau2}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="C5_pencere")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    sel = yaml.safe_load(Path("configs/baselines.yaml").read_text(encoding="utf-8"))
    taus_b = {int(k): v for k, v in sel["tau_inliers"].items()}
    alpha = float(sel["alpha_far_neg"])
    pairs_all = pd.read_csv("data/pairs/pairs_v1.csv", dtype={"region": str, "tile_region": str})
    feats = pd.read_csv("results/faz1/baselines/feats_v1_calib_px5.csv")
    base = B40.evaluate_rows(feats, pairs_all, taus_b)            # satır 1–3 (aynı çiftler)
    pids = sorted(base.pair_id.unique())
    pairs = pairs_all.set_index("pair_id").loc[pids].reset_index()

    tiles = tiles_for(pairs, Path("data/m1/calib_pairs_513.npy"))
    probs, phi_hat = m1_rot_probs(a.model, pairs, tiles)
    taus = train_taus(alpha, taus_b[2])
    print(f"eşikler (eğitim negatifleri, FAR-neg ≤ {alpha}): {taus}; satır 1–3: {taus_b}")

    meta = pairs.set_index("pair_id")
    tile_mpp = meta.tile_size_m / meta.tile_px
    fg = {pid: g.set_index("rot") for pid, g in feats.groupby("pair_id")}
    rows = []
    for k, pid in enumerate(pids):
        g, m = fg[pid], meta.loc[pid]
        order = [ROTS[j] for j in np.argsort(-probs[k], kind="stable")]

        def rec(row, rot, calls, accept, g=g, m=m, pid=pid):
            s = g.loc[rot]
            err = (np.hypot(s.cx - m.gt_x, s.cy - m.gt_y) * tile_mpp[pid]
                   if m.label == "pos" and s.valid and s.n_inliers > 0 else np.nan)
            rows.append({"row": row, "pair_id": pid, "rot": rot, "score": s.n_inliers, "err_m": err,
                         "heading_err": np.nan, "is_pos": m.label == "pos", "neg_type": m.neg_type,
                         "region": m.region, "terrain": m.terrain, "gt_heading": m.gt_heading,
                         "agl_m": m.agl_m, "block": f"{m.region}_{int(pid.split('_')[1]) // B40.BLOCK}",
                         "accept": bool(accept), "calls": calls})

        r = order[0]
        rec("4a", r, 1, g.loc[r].n_inliers >= taus["4a"])
        r = max(order[:2], key=lambda x: g.loc[x].n_inliers)
        rec("4b", r, 2, g.loc[r].n_inliers >= taus["4b"])
        for c, r in enumerate(order, 1):
            if g.loc[r].n_inliers >= taus["4c"]:
                rec("4c", r, c, True)
                break
        else:
            rec("4c", max(order, key=lambda x: g.loc[x].n_inliers), 4, False)
        for c, r in enumerate(order[:2], 1):
            if g.loc[r].n_inliers >= taus["4b"]:
                rec("4d", r, c, True)
                break
        else:
            rec("4d", max(order[:2], key=lambda x: g.loc[x].n_inliers), 2, False)
    r4 = pd.DataFrame(rows)
    base = base.assign(row=base.row.astype(str), calls=base.row.map({1: 1, 2: 4, 3: 4}))
    allr = pd.concat([base, r4], ignore_index=True)

    B = 200 if a.quick else int(cfg["eval"]["bootstrap_B"])
    seed = int(cfg["seed"])
    summ = {}
    for row, g in allr.groupby("row"):
        d = B40.summarize(g, B, seed)
        d["calls_mean"] = float(g.calls.mean())
        d["calls_mean_pos"] = float(g[g.is_pos].calls.mean())
        d["calls_mean_neg"] = float(g[~g.is_pos].calls.mean())
        summ[row] = d
    piv = allr[allr.is_pos].assign(ok=lambda x: x.accept & (x.err_m <= 25)).pivot(
        index="pair_id", columns="row", values="ok")
    comps = [("4a", "1"), ("4d", "1"), ("4d", "2"), ("4c", "2"), ("4a", "2")]
    tests = [S.mcnemar(piv[i].to_numpy(bool), piv[j].to_numpy(bool)) for i, j in comps]
    padj = S.holm([t["p"] for t in tests])
    h1 = yaml.safe_load(Path("configs/h1.yaml").read_text(encoding="utf-8"))
    ok = allr[allr.is_pos].assign(ok=lambda x: x.accept & (x.err_m <= 25))
    okp = ok.pivot(index="pair_id", columns="row", values="ok").astype(float)
    blocks = ok.drop_duplicates("pair_id").set_index("pair_id").loc[okp.index, "block"].to_numpy()
    diffs = {}
    for r4 in ("4c", "4d"):
        dv = (okp[r4] - okp[str(h1["reference_row"])]).to_numpy()
        diffs[r4] = S.cluster_bootstrap_ci(lambda i, dv=dv: dv[i].mean(), blocks,
                                           B=int(h1["bootstrap"]["B"]) if not a.quick else 200, seed=seed)
    pos = pairs[pairs.label == "pos"]
    m1_err = M.angular_error_deg(torch.tensor(phi_hat[pairs.label.to_numpy() == "pos"]),
                                 torch.tensor(pos.gt_heading.to_numpy() % 360, dtype=torch.float32)).numpy()

    crit = h1["criteria"]
    r4 = h1["row"]
    h1res = {"heading_median_deg": float(np.median(m1_err)), "calls_per_pos_frame": summ[r4]["calls_mean_pos"],
             "diff_vs_ref": diffs[r4][0], "diff_ci": [diffs[r4][1], diffs[r4][2]]}
    h1res["i"] = h1res["heading_median_deg"] <= crit["heading_median_deg_max"]
    h1res["ii"] = h1res["calls_per_pos_frame"] <= crit["calls_per_pos_frame_max"]
    h1res["iii"] = diffs[r4][1] > -crit["noninferiority_margin"]
    h1res["H1"] = bool(h1res["i"] and h1res["ii"] and h1res["iii"])
    names = {"1": "RoMa (0°)", "2": "4 yön + en çok inlier", "3": "4 yön + en düşük entropi",
             "4a": "M1-C tek çağrı", "4b": "M1-C ilk 2 dönme", "4c": "M1-C ardışık (τ₂ ile dur)",
             "4d": "M1-C ardışık, en çok 2"}
    tag = f"{a.model}_calib"
    lines = [f"# Satır 4 (M1-C → RoMa) — calib, model `{a.model}`", "",
             (f"Eşikler: satır 1–3 {taus_b}; satır 4 {taus} (eğitim negatifleri, FAR-neg ≤ {alpha}). "
             f"M1-C pozitiflerde medyan yön hatası {np.median(m1_err):.1f}°, < 45° %{100 * (m1_err <= 45).mean():.1f}. "
             f"GA %95, {B} tekrar, {B40.BLOCK} karelik bloklar."), "",
             "| Satır | Yöntem | çağrı/kare (poz/neg) | başarı@25 [GA] | başarı@10 | başarı@50 | AUROC | FAR-neg | FAR-loc |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in ("1", "2", "3", "4a", "4b", "4c", "4d"):
        d = summ[row]
        lines.append(f"| {row} | {names[row]} | {d['calls_mean']:.2f} ({d['calls_mean_pos']:.2f}/"
                     f"{d['calls_mean_neg']:.2f}) | {d['success@25']:.3f} [{d['success@25_ci'][0]:.3f}, "
                     f"{d['success@25_ci'][1]:.3f}] | {d['success@10']:.3f} | {d['success@50']:.3f} | "
                     f"{d['auroc']:.3f} | {d['far_neg']:.3f} | {d['far_loc']:.3f} |")
    lines += ["", "Eşleştirilmiş McNemar (başarı@25, pozitifler), Holm:", "",
              "| Karşılaştırma | yalnız ilki | yalnız ikincisi | p | p (Holm) |", "|---|---|---|---|---|"]
    for (i, j), t, pa in zip(comps, tests, padj, strict=True):
        lines.append(f"| {i} vs {j} | {t['a_only']} | {t['b_only']} | {t['p']:.2e} | {pa:.2e} |")
    lines += ["", f"Başarı@25 farkı (satır − {h1['reference_row']}), blok bootstrap %95 GA: "
              + "; ".join(f"{k}: {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for k, v in diffs.items()), "",
              (f"H1 ön-kaydı (`configs/h1.yaml`, satır {r4}): (i) medyan yön hatası "
              f"{h1res['heading_median_deg']:.1f}° ≤ {crit['heading_median_deg_max']}° → {'✓' if h1res['i'] else '✗'}; "
              f"(ii) çağrı/pozitif kare {h1res['calls_per_pos_frame']:.2f} ≤ {crit['calls_per_pos_frame_max']} → "
              f"{'✓' if h1res['ii'] else '✗'}; (iii) GA alt sınırı {diffs[r4][1]:+.3f} > "
              f"−{crit['noninferiority_margin']} → {'✓' if h1res['iii'] else '✗'}. "
              f"**H1 {'sağlandı' if h1res['H1'] else 'sağlanmadı'}** (calib; karar testte verilir).")]
    lines += ["", "Bölge kırılımı (başarı@25 · çağrı/kare):", "", "| bölge | " + " | ".join(names) + " |",
              "|---|" + "---|" * len(names)]
    for reg, gg in allr[allr.is_pos].groupby("region"):
        cells = []
        for row in names:
            x = gg[gg.row == row]
            cells.append(f"{(x.accept & (x.err_m <= 25)).mean():.3f} · {x.calls.mean():.2f}")
        lines.append(f"| {reg} | " + " | ".join(cells) + " |")
    md = OUT / f"roma_row4_{tag}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    allr.to_csv(OUT / f"roma_row4_decisions_{tag}.csv", index=False)
    (OUT / f"roma_row4_{tag}.json").write_text(json.dumps(
        {"taus": taus, "taus_baseline": taus_b, "summary": summ,
         "mcnemar": [{"a": i, "b": j, **t, "p_holm": float(pa)} for (i, j), t, pa in zip(comps, tests, padj, strict=True)],
         "m1_median_err": float(np.median(m1_err)), "m1_acc45": float((m1_err <= 45).mean()),
         "diff_vs_ref": {k: list(v) for k, v in diffs.items()}, "h1": h1res},
        indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print("\n".join(lines))
    for row in ("4a", "4b", "4c", "4d"):
        for k in ("success@25", "far_neg", "calls_mean"):
            log_run("56_m1c_roma.py", Path("configs/baselines.yaml"), seed, "calib", f"row{row}_{k}",
                    round(float(summ[row][k]), 5), notes=f"model={a.model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
