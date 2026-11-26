# Faz 3 — H2 kuru denemesi (calib 05, 11), `configs/h2.yaml`

S: lojistik, eğitim 41400 (çift, dönme); öznitelikler ['log1p_support', 'support_frac', 'rho', 'spread_m', 'entropy_bits', 'log1p_sum_c', 'frac_dropped']. Eşik eşleme: 5a/5a-x başarı@25'i 4a'ya (0.247), 2v satır 2'ye. Doğru = pozitif ∧ hata ≤ 25 m; yanlış kabul = kabul ∧ ¬doğru.

| satır | yöntem | τ | başarı@25 | yanlış kabul (tüm) | FAR-neg | FAR-loc | AUROC [GA] |
|---|---|---|---|---|---|---|---|
| 4a | M1-C 1 çağrı · RANSAC + inlier | 709 | 0.247 | 0.144 | 0.063 | 0.298 | 0.944 [0.932, 0.955] |
| 5a | M1-C 1 çağrı · hacim + S | -13.3 | 0.233 | 0.920 | 1.000 | 0.767 | 0.936 [0.923, 0.949] |
| 5a-log1p_support |   hacim, yalnız log1p_support | 0.509 | 0.233 | 0.920 | 1.000 | 0.767 | 0.940 [0.928, 0.952] |
| 5a-rho |   hacim, yalnız rho | -1 | 0.233 | 0.920 | 1.000 | 0.767 | 0.912 [0.891, 0.931] |
| 5a-entropy_bits |   hacim, yalnız entropy_bits | -21.1 | 0.233 | 0.920 | 1.000 | 0.767 | 0.920 [0.903, 0.935] |
| 2 | 4 yön · RANSAC + inlier | 1.46e+03 | 0.319 | 0.148 | 0.055 | 0.325 | 0.927 [0.914, 0.940] |
| 2v | 4 yön · hacim + S (en yüksek S) | -10.3 | 0.293 | 0.900 | 1.000 | 0.707 | 0.928 [0.919, 0.938] |

Yanlış kabul McNemar (5a vs 4a): yalnız 4a yanlış kabul 37, yalnız 5a 2441, p = 0.00e+00. 2v vs 2: yalnız 2 44, yalnız 2v 2373, p = 0.00e+00.
AUROC farkı (5a − 4a): -0.008 [-0.021, +0.005].

H2 (`configs/h2.yaml`): (i) yanlış kabul 5a < 4a ve p < 0.05 → ✗; (ii) AUROC farkı GA alt sınırı > 0 → ✗. **H2 sağlanmadı** (calib; karar testte verilir).

S katsayıları (standartlaştırılmış): kesişim -6.43, log1p_support +5.74, support_frac -0.85, rho -1.98, spread_m +0.40, entropy_bits +0.98, log1p_sum_c -3.09, frac_dropped -0.28
