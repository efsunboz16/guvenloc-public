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

## 2. Çift listesi (CSV) — Gün 5–10
`pair_id, drone_path, tile_path, region, split, label(pos/neg), gt_x, gt_y, gt_heading, scale_s, tile_offset_m`

## 3. Olasılık hacmi (M2 → M3, M4) — TBD
## 4. Karar çıktısı (M2/M3 → M4, arayüz) — TBD
`{xy, theta, sigma_hat, score, accept, radius_m}`
## 5. Değerlendirme çağrısı — TBD
`evaluate(pred_csv, split) -> başarı@10/25/50, AUROC, kapsama, yanlış kabul`
