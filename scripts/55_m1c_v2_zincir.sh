#!/bin/bash
# İyileştirme turu: aşama 1 (D1, D2 tohum 0) → val'e göre seçim → aşama 2 (seçilen, tohum 1, 2)
cd "$(dirname "$0")/.." && source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/55_m1c_train.py --veri gercek4 --epochs 12 --batch 32 2>&1 | grep --line-buffered -v apszKnown
BEST=$(python -c "import json; r=json.load(open('results/faz2/m1/m1c_v2.json')); print(min(['D1_olcek','D2_olcek_rot90'], key=lambda k: r[k]['val']['median_err_all']))")
echo "aşama 2: seçilen $BEST (val 12 açı medyanı)"
python scripts/55_m1c_train.py --veri gercek4 --epochs 12 --batch 32 --only $BEST --seeds 1 2 2>&1 | grep --line-buffered -v apszKnown
