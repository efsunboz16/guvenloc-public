"""Gün 1 – ortam kontrolü. Her ekip üyesi çalıştırıp çıktısını paylaşır."""
import importlib
import platform
import sys

import cv2
import numpy
import torch

print("python", sys.version.split()[0], "|", platform.system(), platform.release())
print("torch", torch.__version__, "| cuda", torch.version.cuda)
assert torch.cuda.is_available(), "GPU görünmüyor (CUDA 12.8+ / cu128 PyTorch kurulu mu?)"
p = torch.cuda.get_device_properties(0)
print(f"{p.name} | {p.total_memory / 1e9:.1f} GB | sm_{p.major}{p.minor}")
x = torch.randn(2048, 2048, device="cuda")
print("matmul ok:", bool((x @ x).abs().sum().item() > 0))

for mod in ("romav2", "scipy"):
    importlib.import_module(mod)
print("romav2 import ok | opencv", cv2.__version__, "| numpy", numpy.__version__)
