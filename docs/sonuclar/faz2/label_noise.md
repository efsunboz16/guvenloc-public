# Faz 2A — M1 yön etiketleri ve etiket gürültüsü

Etiket: φ = Phi1 mod 360, 36 bölme (10°). Gürültü: d = θ_RoMa − φ, yalnızca satır 2 kabul + konum hatası < 25 m olan (güvenilir) karelerde. `data/m1/labels_v1.csv` sha256 `a42b77a50a88ffd1…`.

| bölge | küme | kare | bacak | güvenilir | medyan d | medyan \|d\| | p90 \|d\| | \|d\| > 10° | \|d\| > 20° |
|---|---|---|---|---|---|---|---|---|---|
| 01 | train | 817 | 10 | 330 | -9.5° | 9.5° | 18.0° | %48 | %2 |
| 04 | train | 738 | 8 | 161 | -0.2° | 2.2° | 5.2° | %0 | %0 |
| 05 | calib | 473 | 12 | 137 | -1.4° | 3.8° | 8.2° | %4 | %1 |
| 08 | train | 1033 | 10 | 140 | +7.5° | 18.0° | 25.1° | %92 | %44 |
| 09 | train | 766 | 6 | 164 | +2.0° | 2.4° | 3.3° | %1 | %0 |
| 10 | train | 144 | 8 | 7 | +3.1° | 3.2° | 19.6° | %43 | %14 |
| 11 | calib | 590 | 14 | 202 | -1.1° | 1.2° | 4.1° | %0 | %0 |

Eğitim toplam: 802 güvenilir kare, medyan d -3.3°, medyan |d| 5.2°, |d| > 10° %36.

**Doğrulama uçuşu (kural: grup paylaşmayan, ≥ 500 kare, en küçük medyan |d|):** adaylar 01 (9.5°), 04 (2.2°) → **04**. Roller: train 2760, eval 1063, val 738.

Gürültülü bacaklar (≥ 10 güvenilir kare, medyan |d| > 10°): 13
- 01 bacak 1: 82 kare, φ ≈ 27°, medyan d -12.0°, medyan |d| 12.0°
- 01 bacak 3: 82 kare, φ ≈ 31°, medyan d -13.4°, medyan |d| 13.4°
- 01 bacak 5: 83 kare, φ ≈ 29°, medyan d -13.1°, medyan |d| 13.1°
- 01 bacak 7: 84 kare, φ ≈ 30°, medyan d -14.8°, medyan |d| 14.8°
- 01 bacak 9: 86 kare, φ ≈ 32°, medyan d -15.3°, medyan |d| 15.3°
- 08 bacak 2: 101 kare, φ ≈ 110°, medyan d -23.5°, medyan |d| 23.5°
- 08 bacak 3: 97 kare, φ ≈ 278°, medyan d +15.3°, medyan |d| 15.3°
- 08 bacak 4: 100 kare, φ ≈ 109°, medyan d -22.0°, medyan |d| 22.0°
- 08 bacak 5: 100 kare, φ ≈ 278°, medyan d +16.9°, medyan |d| 16.9°
- 08 bacak 6: 104 kare, φ ≈ 112°, medyan d -25.0°, medyan |d| 25.0°
- 08 bacak 7: 105 kare, φ ≈ 278°, medyan d +15.5°, medyan |d| 15.5°
- 08 bacak 8: 108 kare, φ ≈ 109°, medyan d -24.6°, medyan |d| 24.6°
- 08 bacak 9: 108 kare, φ ≈ 279°, medyan d +15.2°, medyan |d| 15.2°
