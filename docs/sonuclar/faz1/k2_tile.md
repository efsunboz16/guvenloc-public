# K2 karo deneyi — eğitim bölgeleri, pozitif çiftler

Bölge başına 60 kare; RANSAC 3 px; GA %95, 2000 tekrar, 50 karelik bloklar. Retsiz (her kare için konum üretilir).

| Satır | Seçenek | karo (m, medyan) | başarı@10 | başarı@25 [GA] | başarı@50 | medyan hata (m) |
|---|---|---|---|---|---|---|
| 1 | fixed | 1400 | 0.027 | 0.100 [0.069, 0.139] | 0.163 | 625.7 |
| 1 | adaptive | 1040 | 0.030 | 0.160 [0.108, 0.223] | 0.267 | 113.6 |
| 1 | fark (uyarlamalı − sabit) | | | +0.060 [+0.022, +0.104] | McNemar 22'e 4, p = 0.00053 | |
| 2 | fixed | 1400 | 0.033 | 0.170 [0.128, 0.218] | 0.273 | 673.5 |
| 2 | adaptive | 1040 | 0.033 | 0.213 [0.160, 0.277] | 0.350 | 323.8 |
| 2 | fark (uyarlamalı − sabit) | | | +0.043 [+0.000, +0.089] | McNemar 22'e 9, p = 0.029 | |

Bölge başına satır 2 başarı@25 (sabit → uyarlamalı): 01: 0.37 → 0.40, 04: 0.13 → 0.25, 08: 0.10 → 0.12, 09: 0.13 → 0.22, 10: 0.12 → 0.08

**Kural sonucu: adaptive** (satır 2 retsiz başarı@25: sabit 0.170, uyarlamalı 0.213).
