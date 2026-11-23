"""Faz 1D – RoMa v2 hız/bellek, kendi kartımızda (gerçek drone↔uydu çifti).

Ölçülen (ısınmadan sonra, --reps tekrar medyanı):
  match_s   : RoMaV2.match (çıkarım)
  sample_s  : dengeli örnekleme — RoMa'nın sample()'ı ve bizim balanced_sample'ımız ayrı
  matcher_s : guvenloc.Matcher.match (örnekleme + Σ + dönme dahil, önbellekte kullanılan)
  peak_gb   : torch.cuda.max_memory_allocated
Kullanım:  python scripts/23_speed.py data/UAV_VisLoc_dataset --settings base fast mega1500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc.io_visloc import crop_square_m, load_region
from guvenloc.matcher import Matcher, balanced_sample
from guvenloc.metrics import log_run


def timed(fn, reps):
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t)
    return float(np.median(out))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--settings", nargs="+", default=["turbo", "fast", "base", "mega1500"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--region", default="04")   # eğitim bölgesi
    args = ap.parse_args()

    reg = load_region(args.root, args.region)
    row = reg.frames.iloc[len(reg.frames) // 2]
    drone = cv2.cvtColor(cv2.imread(str(reg.frame_path(row.filename))), cv2.COLOR_BGR2RGB)
    tile = np.ascontiguousarray(crop_square_m(reg, row.lat, row.lon, 1400, 1024))
    gpu = torch.cuda.get_device_name(0)
    res = []
    for s in args.settings:
        torch.cuda.empty_cache()
        m = Matcher(setting=s)
        reps = 1 if s in ("mega1500", "precise") and "Laptop" in gpu else args.reps
        m.match(drone, tile, 0)  # ısınma
        torch.cuda.reset_peak_memory_stats()
        t_match = timed(lambda m=m: m.model.match(drone, tile), reps)
        preds = m.model.match(drone, tile)
        t_samp_orig = timed(lambda m=m, p=preds: m.model.sample(p, 10_000), reps)
        t_samp_ours = timed(lambda m=m, p=preds: balanced_sample(p, 10_000, m.model.bidirectional), reps)
        t_matcher = timed(lambda m=m: m.match(drone, tile, 90), reps)
        peak = torch.cuda.max_memory_allocated() / 1e9
        r = {"setting": s, "grid": list(m.grid_wh), "bidirectional": bool(m.model.bidirectional),
             "match_s": round(t_match, 3), "sample_romav2_s": round(t_samp_orig, 3),
             "sample_ours_s": round(t_samp_ours, 3), "matcher_s": round(t_matcher, 3),
             "peak_gb": round(peak, 2), "reps": reps}
        res.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        log_run("23_speed.py", Path("configs/base.yaml"), 0, "speed", f"{s}_matcher_s",
                r["matcher_s"], notes=f"{gpu} peak_gb={r['peak_gb']}")
        del m, preds
    out = Path("results/faz1/speed.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gpu": gpu, "torch": torch.__version__, "results": res},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print("rapor:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
