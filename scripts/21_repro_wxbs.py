"""Faz 1D – RoMa v2 tekrarlama: WxBS v1.1 (34 çift).

Hedef (rapor): mAA@10px = 55,4 ± 1 (wxbs ayarı). romav2.benchmarks.wxbs'i sarmalar;
veri ilk koşuda data/WxBS altına otomatik iner (wxbs_benchmark paketi gerekir).

Kullanım:  python scripts/21_repro_wxbs.py [--seeds 0 1 2 3 4]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc.metrics import log_run

TARGET_MAA10 = 55.4


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/WxBS")
    ap.add_argument("--setting", default="wxbs")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args()

    from romav2 import RoMaV2
    from romav2.benchmarks.wxbs import WxBSBenchmark

    torch.set_float32_matmul_precision("highest")
    model = RoMaV2()
    model.apply_setting(a.setting)
    bench = WxBSBenchmark(WxBSBenchmark.Cfg(dataset_path=a.data))
    runs = []
    for s in a.seeds:
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        t0 = time.perf_counter()
        res, thresholds = bench(model, step=0)
        thr = [float(t) for t in np.asarray(thresholds).ravel()]
        res = {k: (np.asarray(v).tolist() if hasattr(v, "__len__") else float(v))
               for k, v in res.items()}
        runs.append({"seed": s, "sec": round(time.perf_counter() - t0, 1), "result": res,
                     "thresholds": thr})
        print(f"tohum {s}: average PCK = {np.round(res['average'], 3).tolist()}")
    # wxbs_benchmark.evaluate_corrs: result["average"] = 0..19 px eşiklerinde PCK (çift ortalaması).
    # mAA@10px = 1..10 px eşiklerinin ortalaması; 0 dahil sürüm de raporlanır (kaynak belirsiz).
    maa, maa0 = [], []
    for r in runs:
        acc = np.asarray(r["result"]["average"], float)
        t = np.asarray(r["thresholds"])
        maa.append(100 * float(acc[(t >= 1) & (t <= 10)].mean()))
        maa0.append(100 * float(acc[t <= 10].mean()))
    summ = {"benchmark": "wxbs_v1.1", "setting": a.setting, "runs": runs,
            "mAA@10px": maa, "mAA@10px_mean": float(np.nanmean(maa)),
            "mAA@10px_incl0": maa0, "mAA@10px_incl0_mean": float(np.nanmean(maa0)),
            "target": TARGET_MAA10, "delta": float(np.nanmean(maa) - TARGET_MAA10)}
    out = Path("results/faz1/repro_wxbs.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"mAA@10px = {summ['mAA@10px_mean']:.2f} (0 dahil: {summ['mAA@10px_incl0_mean']:.2f}) "
          f"(hedef {TARGET_MAA10}, fark {summ['delta']:+.2f})")
    print("rapor:", out)
    log_run("21_repro_wxbs.py", Path("configs/base.yaml"), a.seeds[0], "wxbs", "mAA@10px",
            round(summ["mAA@10px_mean"], 3), notes=f"setting={a.setting} seeds={a.seeds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
