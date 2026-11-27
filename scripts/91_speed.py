"""Faz 6B – hız ve bellek (RTX 3090): modül başına ms ve en yüksek GPU belleği; ablasyon satırları için ms/kare.

  python scripts/91_speed.py --n 60          # calib karelerinden 60 (etiket kullanılmaz; süre etiketten bağımsız)

Ölçülen bileşenler (her biri ısınmadan sonra, CUDA eşitlemeli):
  tile   uydu karosu kırpma (raster okuma)       m1    M1-C birleşimi + TTA (1 kare × 1 karo)
  roma   RoMa v2 eşleştirme (base, 10.000 eşleşme) ransac  homografi + özellikler (CPU)
  vol    oylama hacmi + metrikler (M2)           score S + conformal karar (M3)
  seq    drone↔drone RoMa (M4 odometrisi)         pf    parçacık filtresi tahmin + güncelleme (N = 2.000)
Satır ms/kare = bileşenlerin toplamı × çağrı sayısı (çağrı/kare ablasyon tablosundan, yoksa planlanan değer).
→ results/faz6/speed.json / .md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from guvenloc import baselines as B
from guvenloc import m2_volume as V
from guvenloc import m4_filter as F
from guvenloc import pipeline as P
from guvenloc.io_visloc import crop_square_m, load_region
from guvenloc.matcher import Matcher

DATA = Path("data/UAV_VisLoc_dataset")
OUT = Path("results/faz6")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Timer:
    def __init__(self):
        self.t: dict[str, list[float]] = {}

    def __call__(self, name, fn, *a, **k):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn(*a, **k)
        torch.cuda.synchronize()
        self.t.setdefault(name, []).append(1000 * (time.perf_counter() - t0))
        return r

    def med(self, name) -> float:
        return float(np.median(self.t[name][2:] if len(self.t[name]) > 4 else self.t[name]))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    params = json.loads(Path("results/faz4/m3/m3_params_h3_v1.json").read_text(encoding="utf-8"))
    rc = params["romav2"]
    S67 = _load("m3s67", "scripts/67_m3_pipeline.py")
    pairs = pd.read_csv("data/pairs/pairs_v1.csv", dtype={"region": str, "tile_region": str})
    pos = pairs[(pairs.split == "calib") & (pairs.label == "pos")].sort_values("pair_id")
    pos = pos.iloc[np.linspace(0, len(pos) - 2, a.n).astype(int)]
    torch.cuda.reset_peak_memory_stats()
    matcher = Matcher(setting=rc["setting"], sigma_scale=rc["sigma_scale_matcher"], seed=0)
    m1 = S67.M1Live(params["m1_model"])
    loc = P.Localizer(params, matcher, rc["n_matches"])
    pf = F.ParticleFilter(F.FilterCfg(**json.loads(Path("results/faz5/m4/tune.json").read_text())["selected"]))
    T, regions = Timer(), {}
    for k, r in enumerate(pos.itertuples()):
        if r.tile_region not in regions:
            regions[r.tile_region] = load_region(DATA, r.tile_region)
        reg = regions[r.tile_region]
        rgb = cv2.cvtColor(cv2.imread(str(DATA / r.drone_path)), cv2.COLOR_BGR2RGB)
        bgr4 = cv2.imread(str(DATA / r.drone_path), cv2.IMREAD_REDUCED_COLOR_4)
        tile = T("tile", lambda r=r, reg=reg: np.ascontiguousarray(
            crop_square_m(reg, r.tile_lat, r.tile_lon, r.tile_size_m, int(r.tile_px))))
        t513 = crop_square_m(reg, r.tile_lat, r.tile_lon, r.tile_size_m, 513)
        probs = T("m1", m1, bgr4, t513)
        rot = P.ROTS[int(np.argmax(probs))]
        ms = T("roma", matcher.match, rgb, tile, rot, rc["n_matches"])
        T("ransac", B.features, ms, 5.0)
        mpp = r.tile_size_m / r.tile_px
        vm = T("vol", lambda ms=ms, r=r, mpp=mpp: V.metrics(V.build_volume(
            ms.u, ms.v, ms.c, ms.Sigma_B, ms.Sigma_A, float(r.scale_s), ms.meta["drone_wh"], int(r.tile_px), mpp,
            loc.vcfg)))
        f = vm | {"n_inliers": 1000}
        T("score", loc.decide, f, rot, mpp)
        nxt = pairs[(pairs.region == r.region) & (pairs.label == "pos")
                    & (pairs.pair_id.str.split("_").str[1].astype(int) == int(r.pair_id.split("_")[1]) + 1)]
        if len(nxt):
            rgb2 = cv2.cvtColor(cv2.imread(str(DATA / nxt.drone_path.iloc[0])), cv2.COLOR_BGR2RGB)
            T("seq", matcher.match, rgb, rgb2, 0, rc["n_matches"])
        if not pf.started:
            pf.init(0.0, 0.0, 200.0, 0.0)
        gm = F.GridMeas(np.ones((250, 250), np.float32) / 62500, (0.0, 0.0), int(r.tile_px), mpp, 4.0)
        T("pf", lambda gm=gm: (pf.predict(80.0, 5.0, 0.5), pf.update(gm, 10.0)))
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(pos)}", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    comp = {k: T.med(k) for k in T.t}
    abl = Path("results/faz6/final/ablation_test.json")
    abl = abl if abl.exists() else Path("results/faz6/rehearsal/ablation_calib.json")
    calls = {r: v["calls"] for r, v in json.loads(abl.read_text())["rows"].items()} if abl.exists() else {}
    sat = comp["tile"] + comp["roma"] + comp["ransac"]
    gamma = json.loads(Path("results/faz5/m4/gamma.json").read_text())
    skip = next(x["skip_rate"] for x in gamma["grid"] if x["gamma"] == gamma["selected"])
    rows = {"1": sat, "2": 4 * sat, "3": 4 * sat,
            "4": comp["m1"] + calls.get("4", 1.38) * sat,
            "5": comp["m1"] + sat + comp["vol"],
            "6": comp["m1"] + sat + comp["vol"] + comp["score"],
            "7": comp["m1"] + sat + comp["vol"] + comp["score"] + comp["seq"] + comp["pf"],
            "7-γ": (1 - skip) * (comp["m1"] + sat + comp["vol"] + comp["score"]) + comp["seq"] + comp["pf"]}
    res = {"components_ms": comp, "rows": rows, "peak_gpu_gb": peak, "n": len(pos), "skip_rate_gamma": skip,
           "calls_source": str(abl), "gpu": torch.cuda.get_device_name(0), "setting": rc["setting"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "speed.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    names = {"tile": "uydu karosu kırpma", "m1": "M1-C (3 ağ × 4 TTA)", "roma": "RoMa v2 eşleştirme",
             "ransac": "RANSAC + özellikler (CPU)", "vol": "oylama hacmi + metrikler (M2)", "score": "S + conformal (M3)",
             "seq": "drone↔drone RoMa (M4 odometri)", "pf": "parçacık filtresi adımı (N = 2.000)"}
    L = [f"# Faz 6B — hız ve bellek ({res['gpu']}, RoMa `{rc['setting']}`, {len(pos)} calib karesi)", "",
         "| bileşen | ms (medyan) |", "|---|---|"] + [f"| {names[k]} | {v:.1f} |" for k, v in comp.items()]
    L += ["", "| satır | ms/kare |", "|---|---|"] + [f"| {k} | {v:.0f} |" for k, v in rows.items()]
    L += ["", (f"En yüksek GPU belleği (PyTorch, RoMa + M1-C + M2 aynı süreçte): **{peak:.2f} GB** "
          f"→ 8 GB'lık kartta {'çalışır' if peak < 8 else 'ÇALIŞMAZ'}."),
          (f"M1 çağrı tasarrufu: satır 4 {calls.get('4', float('nan')):.2f} çağrı/kare, satır 2/3: 4 çağrı/kare "
          f"(RoMa başına {sat:.0f} ms → kare başına {4 * sat - rows['4']:.0f} ms kazanç, M1 maliyeti dahil)."),
          (f"γ = {gamma['selected']} m ile uydu yolu karelerin %{100 * skip:.0f}'inde atlanır: satır 7 {rows['7']:.0f} → "
          f"{rows['7-γ']:.0f} ms/kare.")]
    (OUT / "speed.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
