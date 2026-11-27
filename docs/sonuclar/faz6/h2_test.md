# Faz 6 — H2 TEST (02, 03, 06), `configs/h2.yaml`

S: lojistik, eğitim 41400 (çift, dönme); öznitelikler ['log1p_support', 'support_frac', 'rho', 'spread_m', 'entropy_bits', 'log1p_sum_c', 'frac_dropped', 'log1p_inliers']. Eşik eşleme: 5a/5a-x başarı@25'i 4a'ya (0.354), 2v satır 2'ye. Doğru = pozitif ∧ hata ≤ 25 m, hata ortak konumdan (RANSAC merkezi, h2_v3); yanlış kabul = kabul ∧ ¬doğru.

Konum (bilgi, pozitiflerde ≤ 25 m): RANSAC merkezi 0.357, hacim tepesi 0.354.

| satır | yöntem | τ | başarı@25 | yanlış kabul (tüm) | FAR-neg | FAR-loc | AUROC [GA] |
|---|---|---|---|---|---|---|---|
| 4a | M1-C 1 çağrı · RANSAC + inlier | 709 | 0.354 | 0.103 | 0.079 | 0.153 | 0.973 [0.965, 0.980] |
| 5a | M1-C 1 çağrı · hacim + S | -5.63 | 0.355 | 0.165 | 0.142 | 0.212 | 0.975 [0.967, 0.983] |
| 5a-log1p_support |   hacim, yalnız log1p_support | 4.03 | 0.354 | 0.186 | 0.163 | 0.232 | 0.973 [0.964, 0.981] |
| 5a-rho |   hacim, yalnız rho | -0.963 | 0.356 | 0.802 | 0.911 | 0.585 | 0.968 [0.958, 0.978] |
| 5a-entropy_bits |   hacim, yalnız entropy_bits | -18.5 | 0.356 | 0.723 | 0.842 | 0.486 | 0.962 [0.951, 0.971] |
| 2 | 4 yön · RANSAC + inlier | 1.46e+03 | 0.399 | 0.104 | 0.082 | 0.147 | 0.962 [0.954, 0.970] |
| 2v | 4 yön · hacim + S (en yüksek S) | -5.77 | 0.410 | 0.403 | 0.435 | 0.338 | 0.972 [0.963, 0.980] |

Yanlış kabul McNemar (5a vs 4a): yalnız 4a yanlış kabul 177, yalnız 5a 580, p = 8.11e-51. 2v vs 2: yalnız 2 118, yalnız 2v 2069, p = 0.00e+00.
AUROC farkı (5a − 4a): +0.003 [-0.002, +0.007].

H2 (`configs/h2.yaml`): (i) yanlış kabul 5a < 4a ve p < 0.05 → ✗; (ii) AUROC farkı GA alt sınırı > 0 → ✗. **H2 sağlanmadı** (TEST, Faz 6, tek sefer; eşikler calib'den).

S katsayıları (standartlaştırılmış): kesişim -8.03, log1p_support +4.26, support_frac -0.65, rho -2.09, spread_m +0.71, entropy_bits +0.61, log1p_sum_c -2.62, frac_dropped +0.09, log1p_inliers +2.04
