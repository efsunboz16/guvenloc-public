# Taban çizgileri (satır 1–3) — calib, önbellek `dev_base`

RANSAC 3 px (karo pikseli), τ (inlier) = {1: 567, 2: 1251, 3: 294} — eğitimde FAR-neg ≤ 0.05 ile seçildi. Pozitif 1063, negatif 1931 çift. GA: %95, 2000 tekrar, 50 karelik uçuş blokları.

| Satır | Yöntem | çağrı/kare | başarı@10 | başarı@25 [GA] | başarı@50 | başarı@25 (retsiz) | medyan hata (m, kabul) | medyan yön hatası | AUROC [GA] | FAR-neg (zor/kolay) | FAR-loc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | RoMa v2 (0°) + RANSAC | 1 | 0.044 | 0.198 [0.164, 0.232] | 0.406 | 0.202 | 29.4 | 4.2° | 0.950 [0.940, 0.960] | 0.042 (0.063/0.025) | 0.299 |
| 2 | 4 yön + en çok inlier | 4 | 0.070 | 0.291 [0.246, 0.339] | 0.537 | 0.305 | 26.6 | 4.1° | 0.935 [0.922, 0.947] | 0.026 (0.044/0.012) | 0.326 |
| 3 | 4 yön + en düşük entropi | 4 | 0.029 | 0.151 [0.116, 0.189] | 0.294 | 0.153 | 31.7 | 7.0° | 0.968 [0.958, 0.976] | 0.048 (0.069/0.030) | 0.276 |

Eşleştirilmiş McNemar (başarı@25, pozitifler), Holm düzeltmeli:

| Karşılaştırma | yalnız ilki doğru | yalnız ikincisi doğru | p | p (Holm) |
|---|---|---|---|---|
| 2 vs 1 | 106 | 7 | 7.95e-24 | 1.59e-23 |
| 3 vs 1 | 13 | 62 | 8.40e-09 | 8.40e-09 |
| 3 vs 2 | 6 | 154 | 3.02e-38 | 9.05e-38 |

Kırılım — yön dilimi (°) (başarı@25, kabul edilmiş; n):

| yön dilimi (°) | n | satır 1 | satır 2 | satır 3 |
|---|---|---|---|---|
| 000–045 | 1 | 0.000 | 0.000 | 0.000 |
| 045–090 | 143 | 0.154 | 0.252 | 0.147 |
| 090–135 | 390 | 0.203 | 0.326 | 0.182 |
| 225–270 | 342 | 0.196 | 0.240 | 0.108 |
| 270–315 | 187 | 0.225 | 0.342 | 0.171 |

Kırılım — AGL irtifa (m) (başarı@25, kabul edilmiş; n):

| AGL irtifa (m) | n | satır 1 | satır 2 | satır 3 |
|---|---|---|---|---|
| 400–800 | 1062 | 0.198 | 0.291 | 0.152 |
| <400 | 1 | 0.000 | 0.000 | 0.000 |

Kırılım — arazi (başarı@25, kabul edilmiş; n):

| arazi | n | satır 1 | satır 2 | satır 3 |
|---|---|---|---|---|
| tarım | 1063 | 0.198 | 0.291 | 0.151 |

Kırılım — bölge (başarı@25, kabul edilmiş; n):

| bölge | n | satır 1 | satır 2 | satır 3 |
|---|---|---|---|---|
| 05 | 473 | 0.169 | 0.252 | 0.123 |
| 11 | 590 | 0.220 | 0.322 | 0.175 |
