"""GüvenLoc prototipi (MIMARI 6C) — hesap katmanı: tek kare kararı ve uçuş boyunca filtre oynatımı.

Tek kare: `guvenloc.pipeline.Localizer` (h3_v1 satır 6 parametreleri) + M1-C birleşimi (`67_m3_pipeline.M1Live`).
Uçuş: 72_m4_filter.run (tune.json ayarı, γ = gamma.json), kare başına parçacık anlık görüntüleri; ölçümler
results/faz5/m4/meas_<bölge> (71/80). Görüntüler: uydu karosu üzerine tahmin, güven dairesi, GT.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)                      # betikler (56, 67, 72) göreli yollarla depo kökünden çalışır
from guvenloc import m4_filter as F
from guvenloc import pipeline as P
from guvenloc.io_visloc import crop_square_m, load_region

DATA = ROOT / "data/UAV_VisLoc_dataset"
PARAMS = ROOT / "results/faz4/m3/m3_params_h3_v1.json"
R5 = ROOT / "results/faz5/m4"
SPLIT_NAME = {"train": "eğitim", "calib": "kalibrasyon", "test": "test"}
GREEN, RED, BLUE, ORANGE, WHITE = (40, 170, 60), (220, 60, 60), (37, 125, 224), (244, 114, 48), (255, 255, 255)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def pairs() -> pd.DataFrame:
    p = pd.read_csv(ROOT / "data/pairs/pairs_v1.csv", dtype={"region": str, "tile_region": str})
    p = p[p.label == "pos"].copy()
    p["num"] = p.pair_id.str.split("_").str[1].astype(int)
    return p.set_index("pair_id")


@lru_cache(maxsize=12)
def region(rid: str):
    return load_region(DATA, rid)


def frames_of(rid: str) -> list[int]:
    return sorted(pairs()[pairs().region == rid].num.tolist())


def region_choices() -> list[tuple[str, str]]:
    p = pairs()
    out = []
    for rid, g in p.groupby("region"):
        out.append((f"{rid} — {SPLIT_NAME[g.split.iloc[0]]}, {g.terrain.iloc[0]}, {len(g)} kare", rid))
    return out


class Engine:
    """Modeller süreç başına bir kez yüklenir (RoMa v2 + M1-C ≈ 3,6 GB GPU)."""

    def __init__(self):
        from guvenloc.matcher import Matcher
        self.params = json.loads(PARAMS.read_text(encoding="utf-8"))
        rc = self.params["romav2"]
        self.loc = P.Localizer(self.params, Matcher(setting=rc["setting"], sigma_scale=rc["sigma_scale_matcher"],
                                                    seed=0), rc["n_matches"])
        self.m1 = _load("m3s67_app", "scripts/67_m3_pipeline.py").M1Live(self.params["m1_model"])

    def single(self, rid: str, num: int, upload: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, str]:
        r = pairs().loc[f"{rid}_{num:04d}_pos"]
        reg = region(r.tile_region)
        t0 = time.perf_counter()
        if upload is not None:                                      # Gradio: RGB
            rgb = np.ascontiguousarray(upload[..., :3])
            bgr4 = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), None, fx=0.25, fy=0.25,
                              interpolation=cv2.INTER_AREA)
        else:
            rgb = cv2.cvtColor(cv2.imread(str(DATA / r.drone_path)), cv2.COLOR_BGR2RGB)
            bgr4 = cv2.imread(str(DATA / r.drone_path), cv2.IMREAD_REDUCED_COLOR_4)
        tile = np.ascontiguousarray(crop_square_m(reg, r.tile_lat, r.tile_lon, r.tile_size_m, int(r.tile_px))[..., :3])
        probs = self.m1(bgr4, crop_square_m(reg, r.tile_lat, r.tile_lon, r.tile_size_m, 513))
        dcs = self.loc.localize(rgb, tile, float(r.scale_s), float(r.tile_size_m), rot_probs=probs,
                                geo={"tile_lat": r.tile_lat, "tile_lon": r.tile_lon})
        ms = 1000 * (time.perf_counter() - t0)
        mpp = r.tile_size_m / r.tile_px
        img = tile.copy()
        ok = np.isfinite(dcs.xy[0])
        col = GREEN if dcs.accept else RED
        th = max(2, int(r.tile_px / 400))
        gx, gy = int(np.rint(r.gt_x)), int(np.rint(r.gt_y))
        cv2.drawMarker(img, (gx, gy), WHITE, cv2.MARKER_TILTED_CROSS, 18 * th, th + 1)
        if ok:
            x, y = int(np.rint(dcs.xy[0])), int(np.rint(dcs.xy[1]))
            cv2.circle(img, (x, y), max(3, int(dcs.radius_m / mpp)), col, th)
            cv2.circle(img, (x, y), 4 * th, col, -1)
            a = np.radians(dcs.theta)                               # görüntü üstünün yönü (kuzey 0, saat yönü)
            cv2.arrowedLine(img, (x, y), (int(x + 60 * th * np.sin(a)), int(y - 60 * th * np.cos(a))), col, th,
                            tipLength=0.3)
        err = float(np.hypot(dcs.xy[0] - r.gt_x, dcs.xy[1] - r.gt_y) * mpp) if ok else float("nan")
        thumb = cv2.resize(rgb, (640, int(640 * rgb.shape[0] / rgb.shape[1])), interpolation=cv2.INTER_AREA)
        lat, lon = dcs.latlon if dcs.latlon else (float("nan"), float("nan"))
        md = (f"### {'✅ KABUL' if dcs.accept else '⛔ RET'} — {rid} kare {num}"
              f"{' (yüklenen görüntü)' if upload is not None else ''}\n\n"
              f"| | |\n|---|---|\n"
              f"| konum (enlem, boylam) | {lat:.6f}, {lon:.6f} |\n"
              f"| yön (görüntü üstü, pusula) | {dcs.theta:.1f}° |\n"
              f"| güven yarıçapı q̂·σ̂ | {dcs.radius_m:.0f} m (σ̂ = {dcs.sigma_hat:.1f} m) |\n"
              f"| skor S / eşik τ | {dcs.score:.2f} / {self.params['tau']:.2f} |\n"
              f"| M1-C dönme olasılıkları 0/90/180/270° | {' / '.join(f'{p:.2f}' for p in probs)} → {dcs.rot}° |\n"
              f"| inlier | {dcs.n_inliers} |\n"
              f"| hata (GT'ye, bilgi) | {err:.1f} m — GT {'yarıçap içinde' if err <= dcs.radius_m else 'yarıçap dışında'} |\n"
              f"| süre | {ms:.0f} ms |\n\n"
              "Haritada: beyaz × = GT (GPS), nokta + ok = tahmin ve yön, daire = %90 hedefli güven bölgesi "
              "(yeşil kabul, kırmızı ret). Karo, veri kümesindeki pozitif çiftin uydu karosu.")
        return img, thumb, md


# ------------------------------------------------------------------ uçuş (Mod 2)

@lru_cache(maxsize=10)
def flight(rid: str) -> dict:
    S72 = _load("m4s72_app", "scripts/72_m4_filter.py")
    params = json.loads(PARAMS.read_text(encoding="utf-8"))
    cfg = F.FilterCfg(**json.loads((R5 / "tune.json").read_text(encoding="utf-8"))["selected"])
    gamma = float(json.loads((R5 / "gamma.json").read_text(encoding="utf-8"))["selected"])
    kappa = json.loads((R5 / "h4_dry.json").read_text(encoding="utf-8"))["7"]["pooled_in_sample"]["kappa"]
    R = S72.load_region(rid, params, final=True)
    nums = R["d"].num.tolist()
    snaps = dict.fromkeys(nums)
    df = S72.run(R, replace(cfg, seed=0), "7", gamma, snaps)
    d = R["d"]
    return {"df": df.reset_index(drop=True), "snaps": snaps, "kappa": kappa, "gamma": gamma, "nums": nums,
            "lat0": float(d.tile_lat[0]), "lon0": float(d.tile_lon[0]), "paths": d.drone_path.tolist(),
            "tile_region": str(pairs().loc[f"{rid}_{nums[0]:04d}_pos"].tile_region)}


def en_to_latlon(lat0, lon0, e, n):
    lat = lat0 + np.degrees(np.asarray(n) / F.EARTH_R)
    return lat, lon0 + np.degrees(np.asarray(e) / (F.EARTH_R * np.cos(np.radians(lat0))))


def flight_frame(rid: str, k: int, view_m: float = 1500.0, trail: int = 80) -> tuple[np.ndarray, np.ndarray, str]:
    fl = flight(rid)
    df, k = fl["df"], int(np.clip(k, 0, len(fl["nums"]) - 1))
    row = df.iloc[k]
    ce, cn = float(row.e7), float(row.n7)
    lat, lon = en_to_latlon(fl["lat0"], fl["lon0"], ce, cn)
    px = 900
    img = np.ascontiguousarray(crop_square_m(region(fl["tile_region"]), float(lat), float(lon), view_m, px)[..., :3])
    img = (0.75 * img).astype(np.uint8)                            # çizimler okunsun diye karartılmış taban
    s = px / view_m

    def P_(e, n):
        return (int(np.rint(px / 2 + (e - ce) * s)), int(np.rint(px / 2 - (n - cn) * s)))

    lo = max(0, k - trail)
    seg = df.iloc[lo:k + 1]
    pts_gt = np.array([P_(e, n) for e, n in zip(seg["ge"], seg["gn"], strict=True)], np.int32)
    pts_f = np.array([P_(e, n) for e, n in zip(seg.e7, seg.n7, strict=True)], np.int32)
    for i, r in enumerate(seg.itertuples()):                     # tek kare tahminleri (uydu eşleştirmesi yapılanlar)
        if r.matched and np.isfinite(r.e6):
            cv2.circle(img, P_(r.e6, r.n6), 4, ORANGE if r.accept else (150, 110, 90), -1)
        elif not r.matched:
            cv2.circle(img, tuple(pts_f[i]), 3, (180, 180, 180), 1)
    cv2.polylines(img, [pts_gt], False, WHITE, 2, cv2.LINE_AA)
    cv2.polylines(img, [pts_f], False, BLUE, 3, cv2.LINE_AA)
    sn = fl["snaps"].get(fl["nums"][k])
    if sn is not None:
        x, w = sn
        idx = np.random.default_rng(0).choice(len(w), size=min(800, len(w)), replace=False, p=w / w.sum())
        for e, n in x[idx]:
            cv2.circle(img, P_(e, n), 1, (120, 200, 255), -1)
    rad = fl["kappa"] * float(row.spread) if np.isfinite(row.spread) else 0
    cv2.circle(img, P_(ce, cn), max(3, int(rad * s)), BLUE, 2, cv2.LINE_AA)
    cv2.drawMarker(img, P_(row["ge"], row["gn"]), WHITE, cv2.MARKER_TILTED_CROSS, 22, 3)
    cv2.circle(img, P_(ce, cn), 6, BLUE, -1)
    cv2.line(img, (px - 160, px - 30), (px - 160 + int(500 * s), px - 30), WHITE, 3)
    cv2.putText(img, "500 m", (px - 160, px - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, "K", (px - 30, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2, cv2.LINE_AA)
    fr = cv2.imread(str(DATA / fl["paths"][k]), cv2.IMREAD_REDUCED_COLOR_4)
    thumb = cv2.cvtColor(cv2.resize(fr, (480, int(480 * fr.shape[0] / fr.shape[1]))), cv2.COLOR_BGR2RGB)
    upto = df.iloc[:k + 1]
    match_txt = "yapıldı" if row.matched else f"atlandı (yayılım ≤ γ = {fl['gamma']:.0f} m)"
    md = (f"### {rid} — kare {fl['nums'][k]} ({k + 1}/{len(df)})\n\n"
          f"| | bu kare | şimdiye kadar (medyan) |\n|---|---|---|\n"
          f"| filtre hatası | {row.err7:.1f} m | {upto.err7.median():.1f} m |\n"
          f"| tek kare hatası | {row.err6:.1f} m | {upto.err6.median():.1f} m |\n"
          f"| uydu eşleştirmesi | {match_txt} "
          f"| atlanan oran {1 - upto.matched.mean():.0%} |\n"
          f"| M3 kararı | {'kabul' if row.accept else ('ret' if row.matched else '–')} "
          f"| kabul oranı {upto[upto.matched].accept.mean():.0%} (eşleştirilen karelerde) |\n"
          f"| güven yarıçapı κ·yayılım | {rad:.0f} m — GT {'içinde' if row.err7 <= rad else 'dışında'} "
          f"| kapsama {(upto.err7 <= fl['kappa'] * upto.spread).mean():.0%} |\n"
          f"| sıfırlama | {'evet' if row.reset else 'hayır'} | {int(upto.reset.sum())} kez |\n\n"
          "Beyaz çizgi/× = GT (GPS), mavi çizgi/nokta = filtre, açık mavi noktalar = parçacıklar, mavi daire = güven "
          "bölgesi, turuncu = tek kare tahmini (M3 kabul; soluk = ret), gri halka = uydu eşleştirmesi atlanan kare.")
    return img, thumb, md
