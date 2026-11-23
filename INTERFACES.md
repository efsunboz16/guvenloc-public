# GüvenLoc – Modüller arası arayüzler (TASLAK)

Biçim değişikliği PR ile yapılır ve diğer üç üyenin onayını gerektirir.

## 1. Eşleşme dosyası (RoMa önbelleği → M2, M4)  — Gün 1'de doğrulandı

Bir `.npz` dosyası = bir çift (pair_id) × bir döndürme.

| Alan | Şekil | Tip | Açıklama |
|---|---|---|---|
| `u` | (N,2) | float32 | drone karesinde piksel (x, y) |
| `v` | (N,2) | float32 | uydu karosunda piksel (x, y) |
| `c` | (N,) | float16 | RoMa v2 örtüşme (overlap, sigmoid) güveni |
| `P_A` | (N,2,2) | float16 | A (drone) tarafı hassasiyet matrisi |
| `P_B` | (N,2,2) | float16 | B (uydu) tarafı hassasiyet matrisi; Σ = P⁻¹ |
| `meta` | json str | – | pair_id, rot_deg, setting, tile_origin, romav2 sürümü |

N ≈ 10.000 (dengeli örnekleme, `model.sample`).

### RoMa v2 2.0.1 ile ilgili doğrulanan bulgular
- `model.match()` yoğun çıktılar: `warp_AB (1,H,W,2)`, `overlap_AB (1,H,W,1)`, `precision_AB (1,H,W,2,2)`,
  BA yönü için de aynıları (yalnızca çift yönlü ayarlarda). H=W = ayarın yüksek çözünürlüğü.
- `model.sample(preds, N)` → `matches (N,4)` normalize [-1,1] koordinatlarda, `overlaps (N,)`, `prec_A`, `prec_B`.
- **Hata:** tek yönlü ayarlarda (`base`, `fast`, `turbo`) `sample()` hassasiyeti `(N,2)` döndürüyor
  (2×2 matrisin ilk satırı). Çözüm: `precision_AB` yoğun haritasını eşleşme konumunda
  `bhwc_grid_sample` ile okumak (bkz. `scripts/01_roma_single_pair.py`). `matcher.py` bunu sarmalamalı.
- Hassasiyet matrisleri simetrik ve pozitif tanımlı.
- **AÇIK SORU:** Hassasiyetin birimi. Normalize koordinat varsayımıyla hesaplanan σ ≈ 30 px çıkıyor,
  oysa gerçek eşleşme hatası ≈ 0,3 px. `RoMaV2.prec_map_coordinates` ve `geometry.prec_mat_from_prec_params`
  incelenip Σ'nın piksel birimine doğru dönüşümü Gün 2'de netleştirilmeli (Modül 2'nin girdisi).
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

Ek alanlar (geriye uyumlu ekleme — **onay bekliyor**):
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
