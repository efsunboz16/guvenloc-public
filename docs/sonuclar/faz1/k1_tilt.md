# K1 — GT kayması ve kamera eğimi (önceden konmuş kural)

Seçim: satır 2 (RANSAC 5 px, τ₂ = 1462) kabul + hata < 100 m. Model YALNIZCA eğitimde kuruldu. Kayma = RoMa görüntü merkezi − CSV konumu, drone çerçevesinde.

| Model | küme | n | medyan kayma önce (m) | sonra (m) | azalma | %95 GA |
|---|---|---|---|---|---|---|
| metre (kural) | train | 1358 | 21.9 | 16.5 | %24.6 | – |
| metre (kural) | calib | 672 | 24.9 | 19.3 | %22.3 | [14, 28] % |
| AGL'ye bölünmüş (ek bilgi) | train | 1358 | 21.9 | 16.2 | %26.0 | – |
| AGL'ye bölünmüş (ek bilgi) | calib | 672 | 24.9 | 23.3 | %6.5 | [-7, 17] % |

**Katsayılar (kural modeli, eğitim):**

| bileşen | a (m) | b·Omega (m/°) | c·Kappa (m/°) | R² (eğitim) |
|---|---|---|---|---|
| lat_m | 1.77 | -7.00 | 0.97 | 0.582 |
| fwd_m | 6.05 | -1.16 | 5.68 | 0.340 |

**Bölge kırılımı (kural modeli):**

| küme | bölge | n | önce (m) | sonra (m) | azalma | r(yanal, Omega) | r(ileri, Kappa) |
|---|---|---|---|---|---|---|---|
| train | 01 | 470 | 18.6 | 18.1 | %2.9 | -0.71 | 0.69 |
| train | 04 | 454 | 31.1 | 18.4 | %41.0 | -0.86 | 0.94 |
| train | 08 | 201 | 19.0 | 13.0 | %31.5 | -0.73 | 0.47 |
| train | 09 | 218 | 18.0 | 14.9 | %17.3 | -0.70 | 0.50 |
| train | 10 | 15 | 30.2 | 24.7 | %18.2 | 0.09 | 0.20 |
| calib | 05 | 262 | 23.7 | 16.7 | %29.4 | -0.79 | 0.93 |
| calib | 11 | 410 | 25.1 | 21.5 | %14.1 | -0.90 | 0.88 |

**Karar (kural):** calib azalması %22.3 → **konu kapanır (GT = CSV)**.
