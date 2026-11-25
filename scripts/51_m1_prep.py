"""Faz 2C – M1 girdileri: her karenin merkez karesi 321×321 uint8 RGB → data/m1/frames_321.npy.

  python scripts/51_m1_prep.py

Kaynak: data/m1/labels_v2.csv (train/val/eval rolleri; test bölgeleri yok). JPEG 1/4 çözünürlükte çözülür
(cv2.IMREAD_REDUCED_COLOR_4, 994×663), kısa kenar kare kırpılır, 321'e INTER_AREA ile küçültülür.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guvenloc.m1_canon import prep_square

DATA = Path("data/UAV_VisLoc_dataset")
LABELS = Path("data/m1/labels_v2.csv")
OUT = Path("data/m1/frames_321.npy")
SIZE = 321


def load(rel: str) -> np.ndarray:
    im = cv2.imread(str(DATA / rel), cv2.IMREAD_REDUCED_COLOR_4)
    if im is None:
        raise FileNotFoundError(rel)
    return prep_square(im, SIZE)


def main() -> int:
    lab = pd.read_csv(LABELS, dtype={"region": str})
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        frames = list(ex.map(load, lab.drone_path))
    arr = np.stack(frames)
    np.save(OUT, arr)
    print(f"{OUT}: {arr.shape} {arr.nbytes / 1e9:.2f} GB, {time.time() - t0:.0f} s; sıra = {LABELS} satır sırası")
    return 0


if __name__ == "__main__":
    sys.exit(main())
