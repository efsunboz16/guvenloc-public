"""Bölge ayrımı ve çift üretimi (MIMARI.md 1C, INTERFACES.md §2).

Çift türleri:
  pos  — karo merkezi GT'den diskte düzgün kaydırılmış (yarıçap ≤ tile_offset_m)
  hard — aynı bölgede, GT'den ≥ hard_neg_min_dist_m uzakta karo (harita küçükse fallback)
  easy — aynı kümedeki (train/calib/test) BAŞKA gruptan bir bölgede rastgele karo

Karolar kuzey-yukarı ve kare-metriktir; görüntüler burada kesilmez, karo merkezinin
enlem/boylamı ve boyutu saklanır (görüntü `tile_path`'e ayrı adımda kesilir).
Her kare için RNG tohumu (seed, bölge, num) → yeniden üretim birebir, kare eklemek
diğer satırları değiştirmez.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from guvenloc.geo import EARTH_R, GeoRef

SPLITS = ("train", "calib", "test")

PAIR_COLUMNS = [
    # INTERFACES.md §2
    "pair_id", "drone_path", "tile_path", "region", "split", "label",
    "gt_x", "gt_y", "gt_heading", "scale_s", "tile_offset_m",
    # ek alanlar (karo kesimi ve kırılımlar için)
    "neg_type", "tile_region", "tile_lat", "tile_lon", "tile_size_m", "tile_px",
    "drone_lat", "drone_lon", "height", "agl_m", "drone_m_per_px",
    "neg_dist_m", "tile_in_map", "terrain",
]


# ---------------------------------------------------------------------------
# Ayrım
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Split:
    region_split: dict[str, str]          # bölge → train|calib|test
    excluded_regions: frozenset[str]
    excluded_frames: frozenset[str]
    group_of: dict[str, str]              # bölge → grup adı (grupsuz bölge kendi adını alır)
    terrain: dict[str, str]

    def regions(self, split: str) -> list[str]:
        return sorted(r for r, s in self.region_split.items() if s == split)


def load_split(path: str | Path = "configs/split.yaml") -> Split:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    region_split: dict[str, str] = {}
    for s in SPLITS:
        for r in cfg.get(s, []):
            r = str(r).zfill(2)
            if r in region_split:
                raise ValueError(f"{r} birden çok kümede: {region_split[r]}, {s}")
            region_split[r] = s
    excluded = frozenset(str(r).zfill(2) for r in cfg.get("excluded_regions", []))
    if excluded & region_split.keys():
        raise ValueError(f"hem kümede hem dışlanmış: {sorted(excluded & region_split.keys())}")
    group_of = {r: r for r in region_split}
    for g, members in (cfg.get("groups") or {}).items():
        members = [str(m).zfill(2) for m in members]
        if len({region_split.get(m) for m in members}) != 1:
            raise ValueError(f"grup {g} farklı kümelere bölünmüş: "
                             f"{ {m: region_split.get(m) for m in members} }")
        for m in members:
            group_of[m] = g
    terrain = {str(k).zfill(2): v for k, v in (cfg.get("terrain") or {}).items()}
    return Split(region_split, excluded, frozenset(cfg.get("excluded_frames", [])),
                 group_of, terrain)


# ---------------------------------------------------------------------------
# Yerel metrik geometri (bölge ölçeğinde eşdikdörtgen yaklaşım)
# ---------------------------------------------------------------------------
def offset_latlon(lat: float, lon: float, de_m: float, dn_m: float) -> tuple[float, float]:
    """(lat, lon)'dan doğuya de_m, kuzeye dn_m metre öteleme."""
    dlat = math.degrees(dn_m / EARTH_R)
    dlon = math.degrees(de_m / (EARTH_R * math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


def enu_offset_m(lat0: float, lon0: float, lat: float, lon: float) -> tuple[float, float]:
    """(lat0, lon0)'dan (lat, lon)'a (doğu, kuzey) metre."""
    lat_m = math.radians((lat0 + lat) / 2)
    return (math.radians(lon - lon0) * EARTH_R * math.cos(lat_m),
            math.radians(lat - lat0) * EARTH_R)


def tile_in_map_fraction(ref: GeoRef, lat: float, lon: float, size_m: float) -> float:
    """Kenarı size_m olan kuzey-yukarı karonun harita içinde kalan alan oranı."""
    half = size_m / 2
    lat_top, lon_left = offset_latlon(lat, lon, -half, half)
    lat_bot, lon_right = offset_latlon(lat, lon, half, -half)
    w = max(0.0, min(lon_right, ref.lon_right) - max(lon_left, ref.lon_left))
    h = max(0.0, min(lat_top, ref.lat_top) - max(lat_bot, ref.lat_bottom))
    return (w * h) / ((lon_right - lon_left) * (lat_top - lat_bot))


def random_point_in_map(ref: GeoRef, rng: np.random.Generator) -> tuple[float, float]:
    return (float(rng.uniform(ref.lat_bottom, ref.lat_top)),
            float(rng.uniform(ref.lon_left, ref.lon_right)))


# ---------------------------------------------------------------------------
# Çift üretimi
# ---------------------------------------------------------------------------
@dataclass
class RegionInfo:
    region_id: str
    frames: pd.DataFrame
    georef: GeoRef


def drone_m_per_px(height: float, region: str, cfg: dict) -> tuple[float, float]:
    """(AGL metre, drone m/px). Formül configs/pairs.yaml'da."""
    h0 = float((cfg.get("ground_elevation_m") or {}).get(region, 0.0))
    agl = height - h0
    cam = cfg["camera"]
    return agl, cam["footprint_over_agl"] * agl / cam["image_width_px"]


def _sample_far_point(ref, lat, lon, min_dist, cfg, rng):
    for _ in range(int(cfg["max_tries"])):
        p = random_point_in_map(ref, rng)
        d = math.hypot(*enu_offset_m(lat, lon, *p))
        if d >= min_dist and tile_in_map_fraction(ref, *p, cfg["tile_size_m"]) >= cfg[
            "neg_min_tile_in_map"
        ]:
            return p, d
    return None, None


def make_pairs(regions: dict[str, RegionInfo], split: Split, cfg: dict) -> pd.DataFrame:
    """Tüm kümeler için çift tablosu (PAIR_COLUMNS)."""
    seed = int(cfg["seed"])
    size_m, tile_px = float(cfg["tile_size_m"]), int(cfg["tile_px"])
    tile_mpp = size_m / tile_px
    rows = []
    for rid in sorted(regions):
        if rid in split.excluded_regions or rid not in split.region_split:
            continue
        sp = split.region_split[rid]
        info = regions[rid]
        easy_pool = [r for r in split.regions(sp)
                     if r in regions and split.group_of[r] != split.group_of[rid]]
        for _, fr in info.frames.iterrows():
            fname = str(fr["filename"])
            if fname in split.excluded_frames:
                continue
            num = int(fr["num"])
            rng = np.random.default_rng([seed, int(rid), num])
            lat, lon, height = float(fr["lat"]), float(fr["lon"]), float(fr["height"])
            heading = float(fr["Phi1"]) if "Phi1" in fr and pd.notna(fr["Phi1"]) else np.nan
            agl, mpp = drone_m_per_px(height, rid, cfg)
            base = {
                "drone_path": f"{rid}/drone/{fname}", "region": rid, "split": sp,
                "gt_heading": heading, "scale_s": mpp / tile_mpp,
                "tile_size_m": size_m, "tile_px": tile_px,
                "drone_lat": lat, "drone_lon": lon, "height": height, "agl_m": agl,
                "drone_m_per_px": mpp, "terrain": split.terrain.get(rid, ""),
            }
            stem = f"{rid}_{num:04d}"

            # pozitif: diskte düzgün kaydırma
            r = cfg["tile_offset_m"] * math.sqrt(rng.uniform())
            a = rng.uniform(0, 2 * math.pi)
            de, dn = r * math.cos(a), r * math.sin(a)
            tlat, tlon = offset_latlon(lat, lon, de, dn)
            rows.append({**base, "pair_id": f"{stem}_pos", "label": "pos", "neg_type": "pos",
                         "tile_region": rid, "tile_lat": tlat, "tile_lon": tlon,
                         "gt_x": tile_px / 2 - de / tile_mpp, "gt_y": tile_px / 2 + dn / tile_mpp,
                         "tile_offset_m": r, "neg_dist_m": np.nan,
                         "tile_in_map": tile_in_map_fraction(info.georef, tlat, tlon, size_m)})

            # zor negatif: aynı bölge, uzak
            for k in range(int(cfg["hard_neg_per_pos"])):
                p, d = _sample_far_point(info.georef, lat, lon, cfg["hard_neg_min_dist_m"], cfg, rng)
                if p is None:
                    p, d = _sample_far_point(info.georef, lat, lon,
                                             cfg["hard_neg_fallback_dist_m"], cfg, rng)
                if p is None:
                    continue
                rows.append({**base, "pair_id": f"{stem}_hard{k}", "label": "neg",
                             "neg_type": "hard", "tile_region": rid,
                             "tile_lat": p[0], "tile_lon": p[1], "gt_x": np.nan, "gt_y": np.nan,
                             "tile_offset_m": np.nan, "neg_dist_m": d,
                             "tile_in_map": tile_in_map_fraction(info.georef, *p, size_m)})

            # kolay negatif: aynı küme, başka grup
            for k in range(int(cfg["easy_neg_per_pos"])):
                if not easy_pool:
                    break
                other = easy_pool[int(rng.integers(len(easy_pool)))]
                oref = regions[other].georef
                p = None
                for _ in range(int(cfg["max_tries"])):
                    q = random_point_in_map(oref, rng)
                    if tile_in_map_fraction(oref, *q, size_m) >= cfg["neg_min_tile_in_map"]:
                        p = q
                        break
                if p is None:
                    continue
                rows.append({**base, "pair_id": f"{stem}_easy{k}", "label": "neg",
                             "neg_type": "easy", "tile_region": other,
                             "tile_lat": p[0], "tile_lon": p[1], "gt_x": np.nan, "gt_y": np.nan,
                             "tile_offset_m": np.nan, "neg_dist_m": np.nan,
                             "tile_in_map": tile_in_map_fraction(oref, *p, size_m)})
    df = pd.DataFrame(rows)
    df["tile_path"] = "tiles/" + df["pair_id"] + ".jpg"
    return df[PAIR_COLUMNS]
