# Taban çizgileri (satır 1–3) — calib, önbellek `v1`

RANSAC 5 px (karo pikseli), τ (inlier) = {1: 653, 2: 1462, 3: 380} — eğitimde FAR-neg ≤ 0.05 ile seçildi. Pozitif 1063, negatif 2035 çift. GA: %95, 2000 tekrar, 50 karelik uçuş blokları.

| Satır | Yöntem | çağrı/kare | başarı@10 | başarı@25 [GA] | başarı@50 | başarı@25 (retsiz) | medyan hata (m, kabul) | medyan yön hatası | AUROC [GA] | FAR-neg (zor/kolay) | FAR-loc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | RoMa v2 (0°) + RANSAC | 1 | 0.043 | 0.214 [0.176, 0.252] | 0.412 | 0.215 | 27.9 | 4.1° | 0.953 [0.943, 0.963] | 0.068 (0.092/0.046) | 0.285 |
| 2 | 4 yön + en çok inlier | 4 | 0.069 | 0.319 [0.280, 0.361] | 0.575 | 0.335 | 25.1 | 3.8° | 0.927 [0.914, 0.940] | 0.055 (0.084/0.028) | 0.325 |
| 3 | 4 yön + en düşük entropi | 4 | 0.030 | 0.163 [0.124, 0.204] | 0.313 | 0.163 | 30.5 | 5.4° | 0.963 [0.954, 0.972] | 0.063 (0.082/0.045) | 0.260 |

Eşleştirilmiş McNemar (başarı@25, pozitifler), Holm düzeltmeli:

| Karşılaştırma | yalnız ilki doğru | yalnız ikincisi doğru | p | p (Holm) |
|---|---|---|---|---|
| 2 vs 1 | 120 | 8 | 8.99e-27 | 1.80e-26 |
| 3 vs 1 | 10 | 64 | 8.96e-11 | 8.96e-11 |
| 3 vs 2 | 9 | 175 | 4.69e-41 | 1.41e-40 |

Kırılım — yön dilimi (°) (başarı@25, kabul edilmiş; n):

| yön dilimi (°) | n | satır 1 | satır 2 | satır 3 |
|---|---|---|---|---|
| 000–045 | 1 | 0.000 | 0.000 | 0.000 |
| 045–090 | 143 | 0.147 | 0.266 | 0.126 |
| 090–135 | 390 | 0.223 | 0.354 | 0.208 |
| 225–270 | 342 | 0.211 | 0.263 | 0.140 |
| 270–315 | 187 | 0.251 | 0.390 | 0.139 |

Kırılım — AGL irtifa (m) (başarı@25, kabul edilmiş; n):

| AGL irtifa (m) | n | satır 1 | satır 2 | satır 3 |
|---|---|---|---|---|
| 400–800 | 1062 | 0.214 | 0.319 | 0.163 |
| <400 | 1 | 0.000 | 0.000 | 0.000 |

Kırılım — arazi (başarı@25, kabul edilmiş; n):

| arazi | n | satır 1 | satır 2 | satır 3 |
|---|---|---|---|---|
| tarım | 1063 | 0.214 | 0.319 | 0.163 |

Kırılım — bölge (başarı@25, kabul edilmiş; n):

| bölge | n | satır 1 | satır 2 | satır 3 |
|---|---|---|---|---|
| 05 | 473 | 0.192 | 0.290 | 0.144 |
| 11 | 590 | 0.231 | 0.342 | 0.178 |
