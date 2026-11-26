# Faz 4G — uçtan uca hat testi (`guvenloc/pipeline.py`, `scripts/67_m3_pipeline.py`)

100 kare (05 ve 11'den eşit aralıklı), 289 çift (pozitif + kolay/zor negatif). Canlı: M1-C birleşimi (+TTA) → en olası dönme → RoMa v2 → RANSAC + hacim → S → conformal. Çevrimdışı: önbellek (cache/v1) + 64 satır 6. **Değerlendirme değil** (τ, q̂ aynı calib verisinden), tutarlılık testi.

| ölçü | değer |
|---|---|
| dönme uyuşması (canlı M1 vs 4a kararı) | 1.000 |
| kabul uyuşması (canlı dönme) | 1.000 |
| kabul uyuşması (aynı dönme) | 1.000 |
| \|ΔS\| medyan / %95 (aynı dönme) | 0.0000 / 0.0003 |
| n_inliers göreli fark medyan | 0.0000 |
| hacim tepesi kayması medyan / %95 | 0.00 / 0.00 m |
| FAR-neg canlı / çevrimdışı | 0.037 / 0.037 |
| pozitif kabul | 0.490 |
| kapsama (kabul ∧ pozitif) | 0.898 |
| konum hatası medyan (kabul ∧ pozitif) | 33.0 m |
| enlem/boylam ↔ piksel hatası farkı medyan | 0.61 m |
| süre / çift medyan (karo + M1 + RoMa/hacim) | 0.48 s (0.14 + 0.04 + 0.29) |

