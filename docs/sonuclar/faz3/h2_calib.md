# Faz 3 — H2 kuru denemesi (calib 05, 11), `configs/h2.yaml`

S: lojistik, eğitim 41400 (çift, dönme); öznitelikler ['log1p_support', 'support_frac', 'rho', 'spread_m', 'entropy_bits', 'log1p_sum_c', 'frac_dropped']. Eşik eşleme: 5a/5a-x başarı@25'i 4a'ya (0.247), 2v satır 2'ye. Doğru = pozitif ∧ hata ≤ 25 m, hata ortak konumdan (RANSAC merkezi, h2_v2 / F3-8); yanlış kabul = kabul ∧ ¬doğru.

Konum (bilgi, pozitiflerde ≤ 25 m): RANSAC merkezi 0.253, hacim tepesi 0.233. 2v hedefi tavana indirildi (fallback).

| satır | yöntem | τ | başarı@25 | yanlış kabul (tüm) | FAR-neg | FAR-loc | AUROC [GA] |
|---|---|---|---|---|---|---|---|
| 4a | M1-C 1 çağrı · RANSAC + inlier | 709 | 0.247 | 0.144 | 0.063 | 0.298 | 0.944 [0.932, 0.955] |
| 5a | M1-C 1 çağrı · hacim + S | -6.3 | 0.247 | 0.343 | 0.272 | 0.478 | 0.924 [0.909, 0.938] |
| 5a-log1p_support |   hacim, yalnız log1p_support | 4.24 | 0.247 | 0.176 | 0.071 | 0.377 | 0.940 [0.926, 0.952] |
| 5a-rho |   hacim, yalnız rho | -0.954 | 0.247 | 0.779 | 0.833 | 0.675 | 0.857 [0.804, 0.904] |
| 5a-entropy_bits |   hacim, yalnız entropy_bits | -18.4 | 0.247 | 0.607 | 0.619 | 0.583 | 0.905 [0.880, 0.928] |
| 2 | 4 yön · RANSAC + inlier | 1.46e+03 | 0.319 | 0.148 | 0.055 | 0.325 | 0.927 [0.914, 0.940] |
| 2v | 4 yön · hacim + S (en yüksek S) | -7.6 | 0.316 | 0.758 | 0.819 | 0.642 | 0.915 [0.897, 0.933] |

Yanlış kabul McNemar (5a vs 4a): yalnız 4a yanlış kabul 83, yalnız 5a 699, p = 3.39e-122. 2v vs 2: yalnız 2 14, yalnız 2v 1905, p = 0.00e+00.
AUROC farkı (5a − 4a): -0.020 [-0.033, -0.006].

H2 (`configs/h2.yaml`): (i) yanlış kabul 5a < 4a ve p < 0.05 → ✗; (ii) AUROC farkı GA alt sınırı > 0 → ✗. **H2 sağlanmadı** (calib; karar testte verilir).

S katsayıları (standartlaştırılmış): kesişim -6.43, log1p_support +5.74, support_frac -0.85, rho -1.98, spread_m +0.40, entropy_bits +0.98, log1p_sum_c -3.09, frac_dropped -0.28
