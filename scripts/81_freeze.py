"""Faz 6 – test öncesi dondurma: değerlendirmeyi etkileyen her dosyanın sha256'sı → configs/final_manifest.json.

  python scripts/81_freeze.py        # 80_final_prep.py bittikten sonra, 90_ablation.py --final'dan ÖNCE; sonra commit

Dondurulanlar: ön-kayıtlar ve yapılandırmalar (configs/*.yaml), çift listesi, M1-C ağırlıkları ve q (4e), M3 parametreleri,
M4 ayarı/κ/γ, eğitim + calib + test öznitelikleri, test hazırlık çıktıları, değerlendirme betikleri ve paket modülleri.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc import final_guard

ENS = "D2_olcek_rot90+D2_olcek_rot90_s1+D2_olcek_rot90_s2@tta"


def files() -> list[str]:
    f = sorted(str(p) for p in Path("configs").glob("*.yaml"))
    f += ["data/pairs/pairs_v1.csv", "data/m1/labels_v2.csv",
          *(f"cache/models/m1c_{m}.pt" for m in ("D2_olcek_rot90", "D2_olcek_rot90_s1", "D2_olcek_rot90_s2")),
          f"results/faz2/m1/q_sel_{ENS}.json", f"results/faz2/m1/roma_row4_decisions_{ENS}_calib.csv",
          "results/faz4/m3/m3_params_h3_v1.json",
          "results/faz5/m4/tune.json", "results/faz5/m4/h4_dry.json", "results/faz5/m4/gamma.json",
          "results/faz6/prep/m1_probs_test.npz", "results/faz6/prep/seq_motion_test.csv"]
    for s in ("train", "calib", "test"):
        f += [f"results/faz1/baselines/feats_v1_{s}_px5.csv", f"results/faz3/m2/vol_feats_v1_{s}.csv"]
    for r in ("02", "03", "06"):
        f += [f"results/faz5/m4/meas_{r}.npz", f"results/faz5/m4/meas_{r}.csv"]
    f += [f"scripts/{s}" for s in ("40_baselines.py", "55_m1c_train.py", "52_m1_train.py", "56_m1c_roma.py",
                                   "61_m2_score.py", "64_m3_conformal.py", "72_m4_filter.py", "90_ablation.py")]
    f += sorted(str(p) for p in Path("guvenloc").glob("*.py"))
    return f


def main() -> int:
    man = final_guard.freeze(files())
    print(f"dondurildi: {final_guard.MANIFEST} — {len(man['files'])} dosya, commit {man['git_commit'][:8]}")
    print(json.dumps({k: v[:12] for k, v in list(man["files"].items())[:5]}, indent=1), "…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
