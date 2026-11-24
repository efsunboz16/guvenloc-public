"""Faz 1D – RoMa v2 tekrarlama: MegaDepth-1500 ve MegaDepth-C4.

Hedef (rapor): AUC@5/10/20 = 62,8 / 77,0 / 86,6 ± 1 (mega1500 ayarı).
C4: i. çiftin ikinci görüntüsü (i mod 4)·90° döndürülür, K ve poz aynı dönmeyle güncellenir
(guvenloc/repro.py; tests/test_repro.py). Rapor: AUC@5 60,0 → 53,45 düşüşü.

Kullanım (sunucu; 8 GB kartta mega1500 ≈ 50 s/çift — bkz. docs/faz1_tekrarlama.md):
  python scripts/20_repro_mega1500.py --data data/megadepth                    # 1500 çift
  python scripts/20_repro_mega1500.py --data data/megadepth --c4               # C4
  python scripts/20_repro_mega1500.py --data data/megadepth --seeds 0 1 2 3 4  # 5 tohum
  python scripts/20_repro_mega1500.py --data data/megadepth --limit 10 --check-original
Kesintiye dayanıklı: her tohum results/faz1/repro_<ad>_seed<S>.json'a yazılır, var olan tohum
atlanır (--force hariç); özet o ana kadar biten TÜM tohumlardan üretilir. Kaggle'da her oturumda
birkaç tohum koşup tohum dosyalarını sonraki oturuma taşımak yeter (docs/faz1_kaggle.md).
--check-original: aynı alt kümede romav2'nin kendi Mega1500.benchmark'ını da koşar ve
sonucun birebir aynı olduğunu doğrular (C4'ün 0° yolu = orijinal).
Veri: romav2 deposu scripts/eval_prep.sh'in MegaDepth satırları (LoFTR test bölmesi).
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
from guvenloc.repro import load_mega1500, mega1500_eval

TARGET = {"auc@5": 62.8, "auc@10": 77.0, "auc@20": 86.6}


def seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/megadepth")
    ap.add_argument("--setting", default="mega1500")
    ap.add_argument("--c4", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="sahne başına ilk N çift")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--check-original", action="store_true")
    ap.add_argument("--force", action="store_true", help="var olan tohum dosyalarını yeniden hesapla")
    a = ap.parse_args()

    from romav2 import RoMaV2

    torch.set_float32_matmul_precision("highest")
    model = RoMaV2()
    model.apply_setting(a.setting)
    scenes = load_mega1500(a.data, a.limit)
    name = "c4" if a.c4 else "mega1500"
    tag = f"{name}{'_lim' + str(a.limit) if a.limit else ''}"
    res_dir = Path("results/faz1")
    res_dir.mkdir(parents=True, exist_ok=True)
    for s in a.seeds:
        seed_file = res_dir / f"repro_{tag}_seed{s}.json"
        if seed_file.exists() and not a.force and not a.check_original:
            print(f"tohum {s}: zaten var ({seed_file}), atlandı")
            continue
        seed_all(s)
        t0 = time.perf_counter()
        r = mega1500_eval(model, a.data, scenes, c4=a.c4)
        r.update(seed=s, sec=round(time.perf_counter() - t0, 1), setting=a.setting,
                 gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
        tmp = seed_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(seed_file)
        print(f"tohum {s}: AUC@5/10/20 = " + " / ".join(f"{100 * x:.2f}" for x in r["auc"])
              + f"  ({r['n_pairs']} çift, {r['sec']} s)")
        if a.c4:
            for rot, auc in r["auc_by_rot"].items():
                print(f"   {rot:>3}°: " + " / ".join(f"{100 * x:.2f}" for x in auc))

    if a.check_original:
        from romav2.benchmarks.mega1500 import Mega1500

        bench = Mega1500(a.data)
        bench.scenes = scenes
        seed_all(a.seeds[0])
        orig = bench.benchmark(model)
        first = json.loads((res_dir / f"repro_{tag}_seed{a.seeds[0]}.json").read_text("utf-8"))
        ours = first["auc"] if not a.c4 else None
        same = ours is not None and np.allclose(ours, [orig["auc_5"], orig["auc_10"],
                                                       orig["auc_20"]])
        print(f"orijinal Mega1500.benchmark: {orig} → birebir aynı: {same}")

    runs = sorted((json.loads(f.read_text(encoding="utf-8"))
                   for f in res_dir.glob(f"repro_{tag}_seed*.json")), key=lambda r: r["seed"])
    runs = [r for r in runs if r.get("setting", a.setting) == a.setting]
    if not runs:
        print("henüz biten tohum yok")
        return 1
    seeds = [r["seed"] for r in runs]
    auc = np.array([r["auc"] for r in runs]) * 100
    summ = {"benchmark": name, "setting": a.setting, "limit": a.limit, "runs": runs,
            "auc_mean": auc.mean(0).round(2).tolist(), "auc_std": auc.std(0).round(2).tolist(),
            "target": TARGET if not a.c4 else {"auc@5": 53.45}}
    if not a.c4 and a.limit is None:
        summ["delta_vs_target"] = (auc.mean(0) - np.array(list(TARGET.values()))).round(2).tolist()
    print(f"özet ({len(runs)} tohum: {seeds}): AUC@5/10/20 = "
          + " / ".join(f"{m:.2f} ± {sd:.2f}" for m, sd in zip(auc.mean(0), auc.std(0), strict=True)))
    out = res_dir / f"repro_{tag}.json"
    out.write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print("rapor:", out)
    for i, k in enumerate(("auc@5", "auc@10", "auc@20")):
        log_run("20_repro_mega1500.py", Path("configs/base.yaml"), seeds[0], tag, k,
                float(auc.mean(0)[i]),
                notes=f"setting={a.setting} seeds={seeds} gpu={runs[-1].get('gpu', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
