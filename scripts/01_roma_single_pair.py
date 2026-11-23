"""Gün 1 – RoMa v2'yi tek bir çiftte çalıştır, çıktı şekillerini ve hız/belleği kaydet.

Kullanım:
  python scripts/01_roma_single_pair.py                 # yerleşik sentetik çift (bilinen homografi)
  python scripts/01_roma_single_pair.py A.jpg B.jpg     # kendi çiftiniz
  python scripts/01_roma_single_pair.py --setting base  # bellek yetmezse daha düşük çözünürlük

Sentetik çiftte B = H_gt(A) olduğundan kestirilen homografinin hatası da raporlanır.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from romav2 import RoMaV2

OUT = Path("results/day1")


def make_synthetic_pair(out_dir: Path):
    src = Path(matplotlib.get_data_path()) / "sample_data" / "grace_hopper.jpg"
    A = cv2.imread(str(src))
    h, w = A.shape[:2]
    # ~25° dönme + ölçek + perspektif: drone↔uydu farkına kaba bir benzetim
    c = np.array([w / 2, h / 2])
    ang = np.deg2rad(25)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * 0.85
    H = np.eye(3)
    H[:2, :2] = R
    H[:2, 2] = c - R @ c + np.array([15, -10])
    H[2, :2] = [2e-4, -1e-4]
    B = cv2.warpPerspective(A, H, (w, h))
    a, b = out_dir / "synthA.png", out_dir / "synthB.png"
    cv2.imwrite(str(a), A)
    cv2.imwrite(str(b), B)
    return str(a), str(b), H


def to_np(t):
    return t.detach().float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("imgs", nargs="*")
    ap.add_argument("--setting", default="precise",
                    choices=["precise", "base", "fast", "turbo", "mega1500"])
    ap.add_argument("--n", type=int, default=5000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    H_gt = None
    if len(args.imgs) == 2:
        a, b = args.imgs
    else:
        a, b, H_gt = make_synthetic_pair(OUT)

    torch.set_float32_matmul_precision("highest")  # RoMaV2.forward bunu şart koşuyor
    t0 = time.perf_counter()
    model = RoMaV2()
    model.apply_setting(args.setting)
    t_load = time.perf_counter() - t0

    # ısınma + ölçüm
    model.match(a, b)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    preds = model.match(a, b)
    torch.cuda.synchronize()
    t_match = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    matches, overlaps, prec_A, prec_B = model.sample(preds, args.n)
    if prec_B.ndim == 2:
        # romav2 2.0.1 hatası: tek yönlü ayarlarda sample() (N,2,2) yerine (N,2) döndürüyor.
        # Hassasiyeti yoğun precision_AB haritasından, A'daki eşleşme konumunda okuyoruz.
        from romav2.geometry import bhwc_grid_sample
        pm = preds["precision_AB"]  # (1,H,W,2,2)
        h_, w_ = pm.shape[1:3]
        prec_B = bhwc_grid_sample(pm.reshape(1, h_, w_, 4), matches[None, None, :, :2],
                                  mode="bilinear", align_corners=False).reshape(-1, 2, 2)
        prec_A = prec_B  # tek yönlüde ayrı bir A tarafı hassasiyeti yok
    A, B = cv2.imread(a), cv2.imread(b)
    Ha, Wa = A.shape[:2]
    Hb, Wb = B.shape[:2]
    kA, kB = (to_np(k) for k in model.to_pixel_coordinates(matches, Ha, Wa, Hb, Wb))

    shapes = {k: list(v.shape) for k, v in preds.items() if v is not None}
    shapes.update(matches=list(matches.shape), overlaps=list(overlaps.shape),
                  prec_A=list(prec_A.shape), prec_B=list(prec_B.shape))
    for k, v in shapes.items():
        print(f"{k:15s} {v}")

    # hassasiyet → kovaryans (normalize [-1,1] koordinatlarında). Σ = P⁻¹
    P = to_np(prec_B)
    sym_err = float(np.abs(P - P.transpose(0, 2, 1)).max())
    eig = np.linalg.eigvalsh(0.5 * (P + P.transpose(0, 2, 1)))
    Sigma = np.linalg.inv(P)
    # normalize → piksel: x_pix = (x_n+1)/2·W  ⇒  Σ_pix = D Σ D,  D = diag(W/2, H/2)
    D = np.diag([Wb / 2, Hb / 2])
    sigma_px = np.sqrt(np.trace(D @ Sigma @ D, axis1=1, axis2=2) / 2)

    Hest, mask = cv2.findHomography(kA, kB, cv2.USAC_MAGSAC, 3.0, maxIters=10000, confidence=0.999)
    inl = int(mask.sum())
    print(f"inlier: {inl} / {len(kA)}")

    report = {
        "setting": args.setting,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "load_s": round(t_load, 2),
        "match_s": round(t_match, 3),
        "pairs_per_s": round(1 / t_match, 2),
        "peak_mem_gb": round(peak_gb, 2),
        "n_matches": len(kA),
        "inliers": inl,
        "shapes": shapes,
        "precision_sym_err": sym_err,
        "precision_min_eig": float(eig.min()),
        "sigma_px_median": float(np.median(sigma_px)),
        "sigma_px_p90": float(np.percentile(sigma_px, 90)),
    }

    if H_gt is not None:
        pts = np.float32([[0, 0], [Wa, 0], [Wa, Ha], [0, Ha]])[None]
        err = np.linalg.norm(cv2.perspectiveTransform(pts, Hest) - cv2.perspectiveTransform(pts, H_gt), axis=-1)
        proj = cv2.perspectiveTransform(kA[None].astype(np.float64), H_gt)[0]
        match_err = np.linalg.norm(proj - kB, axis=1)
        report.update(H_corner_err_px=float(err.mean()),
                      match_err_px_median=float(np.median(match_err)),
                      match_err_px_p90=float(np.percentile(match_err, 90)),
                      corr_sigma_vs_err=float(np.corrcoef(sigma_px, match_err)[0, 1]))

    canvas = np.zeros((max(Ha, Hb), Wa + Wb, 3), np.uint8)
    canvas[:Ha, :Wa], canvas[:Hb, Wa:] = A, B
    idx = np.flatnonzero(mask.ravel())
    for i in np.random.default_rng(0).choice(idx, min(150, len(idx)), replace=False):
        cv2.line(canvas, (int(kA[i, 0]), int(kA[i, 1])), (int(kB[i, 0]) + Wa, int(kB[i, 1])),
                 (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "pair_out.jpg"), canvas)

    np.savez_compressed(OUT / "sample_matches.npz", u=kA.astype(np.float32), v=kB.astype(np.float32),
                        c=to_np(overlaps).astype(np.float16), P_A=to_np(prec_A).astype(np.float16),
                        P_B=P.astype(np.float16))
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "shapes"}, indent=2))


if __name__ == "__main__":
    main()
