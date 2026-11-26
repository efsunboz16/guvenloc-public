# Faz 3 — H2 kuru denemesi (calib 05, 11), `configs/h2.yaml`

S: lojistik, eğitim 41400 (çift, dönme); öznitelikler ['log1p_support', 'support_frac', 'rho', 'spread_m', 'entropy_bits', 'log1p_sum_c', 'frac_dropped', 'log1p_inliers']. Eşik eşleme: 5a/5a-x başarı@25'i 4a'ya (0.249), 2v satır 2'ye. Doğru = pozitif ∧ hata ≤ 25 m, hata ortak konumdan (RANSAC merkezi, h2_v3); yanlış kabul = kabul ∧ ¬doğru.

Konum (bilgi, pozitiflerde ≤ 25 m): RANSAC merkezi 0.254, hacim tepesi 0.237.

| satır | yöntem | τ | başarı@25 | yanlış kabul (tüm) | FAR-neg | FAR-loc | AUROC [GA] |
|---|---|---|---|---|---|---|---|
| 4a | M1-C 1 çağrı · RANSAC + inlier | 709 | 0.249 | 0.142 | 0.062 | 0.294 | 0.945 [0.933, 0.956] |
| 5a | M1-C 1 çağrı · hacim + S | -5.63 | 0.249 | 0.185 | 0.088 | 0.370 | 0.937 [0.927, 0.947] |
| 5a-log1p_support |   hacim, yalnız log1p_support | 4.03 | 0.249 | 0.202 | 0.097 | 0.404 | 0.941 [0.928, 0.953] |
| 5a-rho |   hacim, yalnız rho | -0.963 | 0.249 | 0.787 | 0.846 | 0.675 | 0.865 [0.813, 0.912] |
| 5a-entropy_bits |   hacim, yalnız entropy_bits | -18.5 | 0.249 | 0.627 | 0.645 | 0.592 | 0.907 [0.881, 0.930] |
| 2 | 4 yön · RANSAC + inlier | 1.46e+03 | 0.319 | 0.148 | 0.055 | 0.325 | 0.927 [0.914, 0.940] |
| 2v | 4 yön · hacim + S (en yüksek S) | -5.77 | 0.319 | 0.358 | 0.280 | 0.508 | 0.921 [0.908, 0.934] |

Yanlış kabul McNemar (5a vs 4a): yalnız 4a yanlış kabul 82, yalnız 5a 216, p = 4.50e-15. 2v vs 2: yalnız 2 32, yalnız 2v 683, p = 4.98e-160.
AUROC farkı (5a − 4a): -0.008 [-0.016, +0.001].

H2 (`configs/h2.yaml`): (i) yanlış kabul 5a < 4a ve p < 0.05 → ✗; (ii) AUROC farkı GA alt sınırı > 0 → ✗. **H2 sağlanmadı** (calib; karar testte verilir).

S katsayıları (standartlaştırılmış): kesişim -8.03, log1p_support +4.26, support_frac -0.65, rho -2.09, spread_m +0.71, entropy_bits +0.61, log1p_sum_c -2.62, frac_dropped +0.09, log1p_inliers +2.04
