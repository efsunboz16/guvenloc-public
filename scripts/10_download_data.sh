#!/usr/bin/env bash
# Veri indirme yönergesi (MIMARI.md 1A). Bağlantılar README'lerden doğrulanarak doldurulur.
set -euo pipefail
mkdir -p data/uav_visloc_sample data/uav_visloc data/megadepth data/WxBS
echo "1) UAV-VisLoc: https://github.com/IntelliSensing/UAV-VisLoc README'sindeki Google Drive"
echo "   bağlantısından 16,4 GB sürümü indirin (Zenodo 197 GB sürümü DEĞİL) -> data/uav_visloc/"
echo "2) MegaDepth-1500: romav2 deposu scripts/eval_prep.sh (yalnızca MegaDepth satırları) -> data/megadepth/"
echo "3) WxBS: romav2.benchmarks.wxbs otomatik indirir (wxbs_benchmark paketi gerekir)"
