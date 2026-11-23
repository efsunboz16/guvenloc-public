"""Faz 1A – UAV-VisLoc indirmesini doğrula.

Kullanım:
  python scripts/11_verify_data.py data/uav_visloc_sample
  python scripts/11_verify_data.py data/uav_visloc --expect-frames 6742 --expect-maps 11 --checksums

Kontroller:
  * bölge klasörleri (NN/), her birinde drone/, satelliteNN.tif, NN.csv
  * drone karesi sayısı ↔ CSV satır sayısı ↔ CSV'de adı geçen dosyalar
  * CSV sütun adları ve her sütunun boş/eksik oranı (yön sütunu hangi bölgede yok?)
  * uydu haritası boyutu, CRS ve sınırları (rasterio) + satellite_coordinates_range.csv
  * isteğe bağlı: sha256 listesi -> docs/checksums/<klasör>.sha256
Sonuç ayrıca results/faz1/verify_<klasör>.json olarak yazılır.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_csv_any(path: Path) -> pd.DataFrame:
    # Ayırıcı belirsiz (virgül / boşluk / sekme) — pandas'a tahmin ettir.
    return pd.read_csv(path, sep=None, engine="python")


def bounds_diff_m(csv_box, ras_box) -> list[float]:
    """Koordinat CSV'si ile raster sınırları arasındaki fark (m): [sol, üst, sağ, alt]."""
    import math

    lat_t, lon_l, lat_b, lon_r = csv_box
    left, bottom, right, top = ras_box
    k = 111_320.0
    c = math.cos(math.radians((lat_t + lat_b) / 2))
    return [round((lon_l - left) * k * c, 2), round((lat_t - top) * k, 2),
            round((lon_r - right) * k * c, 2), round((lat_b - bottom) * k, 2)]


def raster_info(path: Path) -> dict:
    try:
        import rasterio
        from rasterio.errors import RasterioError
    except ImportError:
        return {"error": "rasterio kurulu değil"}
    try:
        with rasterio.open(path) as ds:
            return {
                "width": ds.width,
                "height": ds.height,
                "bands": ds.count,
                "dtype": ds.dtypes[0],
                "crs": str(ds.crs) if ds.crs else None,
                "bounds": list(ds.bounds),
                "res": list(ds.res),
            }
    except (RasterioError, OSError) as e:  # bozuk / eksik tif
        return {"error": repr(e)}


def check_region(rdir: Path) -> dict:
    rid = rdir.name
    out: dict = {"region": rid, "problems": []}
    drone_dir = rdir / "drone"
    frames = sorted(p for p in drone_dir.glob("*") if p.suffix.lower() in IMG_EXT) if drone_dir.is_dir() else []
    out["n_frames"] = len(frames)
    if not frames:
        out["problems"].append("drone/ klasörü yok veya boş")

    tifs = sorted(rdir.glob("satellite*.tif"))
    out["satellite"] = [t.name for t in tifs]
    out["raster"] = None
    if not tifs:
        out["problems"].append("uydu tif'i yok")
    elif len(tifs) == 1:
        out["raster"] = raster_info(tifs[0])
    else:
        # Parçalı harita (tam pakette 09): VRT mozaiği tek harita olarak doğrulanır.
        from guvenloc.io_visloc import satellite_source

        try:
            out["raster"] = raster_info(satellite_source(rdir))
            out["raster"]["tiles"] = len(tifs)
        except (ValueError, OSError) as e:
            out["problems"].append(f"{len(tifs)} parçalı harita birleştirilemedi: {e!r}")
    if out["raster"] and "error" in out["raster"]:
        out["problems"].append(f"tif okunamadı: {out['raster']['error']}")

    csvs = sorted(rdir.glob("*.csv"))
    out["csv"] = [c.name for c in csvs]
    if len(csvs) != 1:
        out["problems"].append(f"{len(csvs)} CSV bulundu (1 bekleniyor)")
        return out

    df = read_csv_any(csvs[0])
    out["n_rows"] = len(df)
    out["columns"] = list(map(str, df.columns))
    out["missing_ratio"] = {c: round(float(df[c].isna().mean()), 4) for c in df.columns}
    out["head"] = df.head(3).astype(str).to_dict(orient="records")
    num = df.select_dtypes("number")
    out["numeric_range"] = {c: [float(num[c].min()), float(num[c].max())] for c in num.columns}
    out["has_heading"] = "Phi1" in df.columns and bool(df["Phi1"].notna().any())

    ras = out["raster"] or {}
    if "bounds" in ras and {"lat", "lon"} <= set(df.columns):
        left, bottom, right, top = ras["bounds"]
        outside = (df.lat > top) | (df.lat < bottom) | (df.lon < left) | (df.lon > right)
        out["frames_outside_map"] = int(outside.sum())
        if outside.any():
            out["outside_map_files"] = df.loc[outside, df.columns[1]].astype(str).tolist()[:20]

    # CSV'deki dosya adları ile diskteki kareleri eşle
    name_col = next((c for c in df.columns if re.search(r"file|name|img", str(c), re.IGNORECASE)), df.columns[0])
    listed = {Path(str(v)).name.lower() for v in df[name_col].dropna()}
    on_disk = {p.name.lower() for p in frames}
    out["name_column"] = str(name_col)
    out["csv_not_on_disk"] = sorted(listed - on_disk)[:20]
    out["disk_not_in_csv"] = sorted(on_disk - listed)[:20]
    if listed != on_disk:
        out["problems"].append(
            f"CSV↔disk uyuşmazlığı: {len(listed - on_disk)} CSV'de var/diskte yok, "
            f"{len(on_disk - listed)} diskte var/CSV'de yok"
        )
    return out


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Windows konsolu (cp1252) Türkçe karakterde çöker
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--expect-frames", type=int)
    ap.add_argument("--expect-maps", type=int)
    ap.add_argument("--checksums", action="store_true")
    args = ap.parse_args()

    root: Path = args.root
    # Zip bazen tek bir üst klasörle açılır (UAV-VisLoc/…): bölge klasörlerini bulana kadar in.
    while root.is_dir() and not any(p.is_dir() and re.fullmatch(r"\d{2}", p.name) for p in root.iterdir()):
        subdirs = [p for p in root.iterdir() if p.is_dir()]
        if len(subdirs) != 1:
            break
        root = subdirs[0]
    regions = sorted(p for p in root.iterdir() if p.is_dir() and re.fullmatch(r"\d{2}", p.name)) if root.is_dir() else []
    if not regions:
        print(f"HATA: {args.root} altında NN/ biçiminde bölge klasörü bulunamadı")
        return 1

    report: dict = {"root": str(root), "regions": [check_region(r) for r in regions]}
    # Örnek pakette dosya adı "satellite_ coordinates_range.csv" (boşluklu) — esnek ara.
    rng = next(iter(sorted(root.glob("satellite*coordinates*range*.csv"))), None)
    if rng is not None:
        report["satellite_coordinates_range_file"] = rng.name
        rdf = read_csv_any(rng)
        report["satellite_coordinates_range"] = rdf.astype(str).to_dict(orient="records")
        cols = {"LT_lat_map", "LT_lon_map", "RB_lat_map", "RB_lon_map", "mapname"}
        if cols <= set(rdf.columns):
            for _, row in rdf.iterrows():
                rid = re.sub(r"\D", "", str(row["mapname"])).zfill(2)
                reg = next((r for r in report["regions"] if r["region"] == rid), None)
                if reg and reg["raster"] and "bounds" in reg["raster"]:
                    box = (row.LT_lat_map, row.LT_lon_map, row.RB_lat_map, row.RB_lon_map)
                    reg["range_csv_vs_raster_m"] = bounds_diff_m(box, reg["raster"]["bounds"])
    else:
        report["satellite_coordinates_range"] = None

    n_frames = sum(r["n_frames"] for r in report["regions"])
    n_maps = sum(1 for r in report["regions"] if r["satellite"])
    report["totals"] = {"regions": len(regions), "frames": n_frames, "maps": n_maps}

    problems = [f"{r['region']}: {p}" for r in report["regions"] for p in r["problems"]]
    if args.expect_frames is not None and n_frames != args.expect_frames:
        problems.append(f"toplam kare {n_frames} ≠ beklenen {args.expect_frames}")
    if args.expect_maps is not None and n_maps != args.expect_maps:
        problems.append(f"toplam harita {n_maps} ≠ beklenen {args.expect_maps}")
    if report["satellite_coordinates_range"] is None:
        problems.append("satellite_coordinates_range.csv bulunamadı")
    report["problems"] = problems

    # ---- özet çıktı
    print(f"kök: {root}")
    print(f"bölge: {len(regions)} | kare: {n_frames} | harita: {n_maps}")
    for r in report["regions"]:
        ras = r.get("raster") or {}
        size = f"{ras.get('width')}x{ras.get('height')}" if "width" in ras else "-"
        extra = []
        if ras.get("tiles"):
            extra.append(f"{ras['tiles']} parça→VRT")
        if not r.get("has_heading", True):
            extra.append("YÖN YOK")
        if r.get("frames_outside_map"):
            extra.append(f"{r['frames_outside_map']} kare harita dışı")
        d = r.get("range_csv_vs_raster_m")
        if d and max(map(abs, d)) > 1:
            extra.append(f"CSV↔tif sınır farkı {d} m")
        print(f"  {r['region']}: {r['n_frames']:5d} kare, CSV {r.get('n_rows', '-')} satır, "
              f"harita {size}, CRS {ras.get('crs')}" + (f"  [{'; '.join(extra)}]" if extra else ""))
    cols = {tuple(r.get("columns", [])) for r in report["regions"]}
    print("CSV sütunları:", *[list(c) for c in cols], sep="\n  ")
    for r in report["regions"]:
        miss = {c: v for c, v in r.get("missing_ratio", {}).items() if v > 0}
        if miss:
            print(f"  {r['region']} eksik değerler: {miss}")
    print("SORUNLAR:" if problems else "Sorun bulunmadı.", *problems, sep="\n  ")

    out_dir = Path("results/faz1")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"verify_{args.root.name}.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"rapor: {out_json}")

    if args.checksums:
        ck_dir = Path("docs/checksums")
        ck_dir.mkdir(parents=True, exist_ok=True)
        ck = ck_dir / f"{args.root.name}.sha256"
        # Türetilmiş VRT mozaikleri listeye girmez: checksum'lar yalnızca indirilen dosyalar.
        files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.endswith("_mosaic.vrt"))
        with ck.open("w", encoding="utf-8", newline="\n") as f:
            for i, p in enumerate(files, 1):
                f.write(f"{sha256(p)}  {p.relative_to(root).as_posix()}\n")
                if i % 500 == 0:
                    print(f"  sha256 {i}/{len(files)}")
        print(f"checksum: {ck} ({len(files)} dosya)")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
