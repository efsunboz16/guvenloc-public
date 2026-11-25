# GüvenLoc – Modüller arası arayüzler (§1–2 onaylı, 25.11)

Biçim değişikliği PR ile yapılır; gerekçesi PR açıklamasına yazılır (tek geliştirici, MIMARI §3.1).

## 1. Eşleşme dosyası (RoMa önbelleği → M2, M4) — Faz 1E'de uygulandı

Yazan: `guvenloc.matcher.MatchSet.save_npz`; doğrulayan: `guvenloc.cache.validate_npz`;
üreten: `scripts/30_build_cache.py`. Bir `.npz` = bir çift × bir yön:
`cache/v1/<pair_id>_r<rot>.npz`, ardışık kareler `cache/v1/seq/<bölge>_<t>_<t+1>.npz`.

| Alan | Şekil | Tip | Açıklama |
|---|---|---|---|
| `u` | (N,2) | float32 | drone karesinde piksel (x, y), **orijinal (döndürülmemiş) kare**, OpenCV kuralı (piksel merkezi tamsayı) |
| `v` | (N,2) | float32 | uydu karosunda piksel (x, y), aynı kural |
| `c` | (N,) | float16 | RoMa v2 örtüşme güveni (sigmoid) |
| `Sigma_A` | (N,2,2) | float32 | `u`'nun kovaryansı, **drone pikseli²**; tek yönlü ayarlarda 0 (A ucu ızgara noktası) |
| `Sigma_B` | (N,2,2) | float32 | `v`'nin kovaryansı, **karo pikseli²** |
| `meta` | json str | – | setting, romav2, rot_deg, drone_wh, sat_wh, grid_wh, sigma_scale, bidirectional, coords, cov_units, sigma_eig_range_px2, kind, pair_id, label, neg_type, split |

Taslaktan değişenler (onaylandı, 25.11 — faz1_durum K3):
- `P_A`/`P_B` (hassasiyet, fp16) yerine `Sigma_A`/`Sigma_B` (kovaryans, float32). fp16 hem taşıyordu
  (σ > 256 px) hem de anizotrop matrislerde pozitif tanımlılığı bozuyordu. Özdeğerler [1e-4, 1e6] px².
- `rot` (dosya adı) ve `meta.rot_deg`: drone karesine **saat yönünde** uygulanan dönme
  (`geo.north_up_rotation_cw_deg` ile aynı kural); `u` her zaman orijinal kare koordinatında.

**Σ ölçeği (`sigma_scale`, 26.11):** `cache/v1` ham Σ saklar (`meta.sigma_scale = 1.0`). Kalibre ölçek
`configs/base.yaml: romav2.sigma_scale` (= 0,9272, `scripts/42_sigma_scale.py`). Tüketici (M2–M4) Σ'yı
`(config.sigma_scale / meta.sigma_scale)²` ile çarpar; böylece config değerle üretilmiş yeni bir önbellek iki kez
ölçeklenmez. `base` önbelleği tek yönlüdür: `Sigma_A = 0` (yalnızca `Sigma_B` anlamlı).

N ≈ 10.000: `matcher.balanced_sample` — RoMaV2.sample() ile birebir aynı örnekleme (aynı tohumla
aynı eşleşmeler doğrulandı), yalnızca KDE parça parça (8 GB kartta 1,7 s → ~0,1 s).

### Hassasiyet birimi — Faz 0.4'te ÇÖZÜLDÜ (`scripts/02_precision_units.py`)
RoMa v2 `precision_AB`, refiner'ın **son aşamasının ızgara pikselinde** tanımlı (hr varsa hr, yoksa lr:
`base` → 640, `mega1500` → 1024). Normalize koordinat değil: o varsayım σ'yı ~1000 kat büyük veriyordu
(Gün 1'deki "σ ≈ 30 px, hata ≈ 0,3 px" çelişkisi). Görüntü pikseline: `P_img = D P D`,
`D = diag(W_ızgara/W_img, H_ızgara/H_img)` (= `RoMaV2.prec_map_coordinates`). Ayrıntı: `docs/faz1_matcher_onbellek.md`.

### RoMa v2 2.0.1 ile ilgili doğrulanan bulgular
- `model.match()` yoğun çıktılar: `warp_AB (1,H,W,2)`, `overlap_AB (1,H,W,1)`, `precision_AB (1,H,W,2,2)`,
  BA yönü için de aynıları (yalnızca çift yönlü ayarlarda). H=W = ayarın son çözünürlüğü.
- **Hata:** tek yönlü ayarlarda (`base`, `fast`, `turbo`) `sample()` hassasiyeti `(N,2)` döndürüyor.
  `matcher.py` hassasiyeti yoğun haritalardan okur.
- `forward` için `torch.set_float32_matmul_precision("highest")` şart.

## 2. Çift listesi (CSV) — Faz 1C'de uygulandı (`guvenloc/pairs.py`, `scripts/14_make_pairs.py`)
`data/pairs/pairs_v1.csv` (git dışı; sha256 `docs/checksums/pairs_v1.sha256`, aynı seed ile birebir yeniden üretilir).

Temel alanlar (taslaktaki gibi):
`pair_id, drone_path, tile_path, region, split, label(pos/neg), gt_x, gt_y, gt_heading, scale_s, tile_offset_m`

| Alan | Tanım |
|---|---|
| `pair_id` | `<bölge>_<num:04d>_{pos, hard0, easy0}` |
| `drone_path` | veri köküne göre: `NN/drone/NN_XXXX.JPG` |
| `tile_path` | `data/pairs/`'e göre: `tiles/<pair_id>.jpg` (`--cut-tiles` ile kesilir) |
| `gt_x, gt_y` | GT drone merkezi, **karo pikseli** (kuzey-yukarı, `tile_px` × `tile_px`); negatiflerde boş |
| `gt_heading` | Phi1: görüntü üst kenarının pusula yönü (kuzey 0, saat yönü), derece |
| `scale_s` | drone pikseli → karo pikseli ölçek oranı = `drone_m_per_px / (tile_size_m / tile_px)` |
| `tile_offset_m` | pozitiflerde karo merkezinin GT'den uzaklığı (m) |

Ek alanlar (geriye uyumlu ekleme — onaylandı, 25.11):
`neg_type (pos/hard/easy), tile_region, tile_lat, tile_lon, tile_size_m, tile_px, drone_lat, drone_lon,
height, agl_m, drone_m_per_px, neg_dist_m, tile_in_map, terrain`
- `tile_region`: karonun kesildiği bölge (kolay negatifte başka bölge).
- `tile_lat, tile_lon`: karo merkezi; karo = `io_visloc.crop_square_m(tile_region, tile_lat, tile_lon, tile_size_m, tile_px)`.
- `tile_in_map`: karonun harita içinde kalan alan oranı (harita kenarında < 1; dışı siyah).

## 3. Olasılık hacmi (M2 → M3, M4) — TBD
## 4. Karar çıktısı (M2/M3 → M4, arayüz) — TBD
`{xy, theta, sigma_hat, score, accept, radius_m}`
## 5. Değerlendirme çağrısı — TBD
`evaluate(pred_csv, split) -> başarı@10/25/50, AUROC, kapsama, yanlış kabul`
