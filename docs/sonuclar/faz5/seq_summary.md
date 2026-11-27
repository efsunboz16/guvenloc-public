# Faz 5A — uçuş dizileri (eğitim + calib; test kapalı)

Ardışık pozitif kareler. Hareket hatası = |homografi yer değiştirmesi − GPS yer değiştirmesi| (m), Δθ hatası = |homografi yön farkı − ΔΦ1| (°). 'kullanılır' = homografi inlier ≥ 50.

| bölge | küme | çift | atlama > 1 | mesafe medyan | Δt medyan | örtüşme medyan | örtüşmesiz | inlier medyan | kullanılır | hareket hatası medyan / %90 | Δθ hatası medyan |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 01 | train | 816 | 0 | 79 m | 4 s | 0.70 | 0.000 | 3360 | 1.000 | 23.8 / 52.5 m | 1.11° |
| 04 | train | 737 | 0 | 111 m | 5 s | 0.69 | 0.000 | 7897 | 1.000 | 28.6 / 60.9 m | 0.75° |
| 05 | calib | 472 | 0 | 60 m | 3 s | 0.82 | 0.002 | 6723 | 0.998 | 26.4 / 53.5 m | 0.54° |
| 08 | train | 1032 | 1 | 100 m | 4 s | 0.72 | 0.000 | 7528 | 1.000 | 35.2 / 63.3 m | 0.50° |
| 09 | train | 765 | 0 | 100 m | 5 s | 0.72 | 0.000 | 7375 | 1.000 | 11.8 / 26.3 m | 0.17° |
| 10 | train | 143 | 0 | 60 m | 3 s | 0.76 | 0.000 | 9136 | 1.000 | 14.7 / 31.2 m | 1.18° |
| 11 | calib | 589 | 0 | 130 m | 7 s | 0.75 | 0.003 | 8966 | 0.997 | 36.5 / 71.6 m | 0.38° |

Toplam 4554 çift; kullanılır 0.999. Kullanılır çiftlerde hareket hatası medyan 25.5 m (%90 59.1 m), mesafeye oranı medyan 0.273; Δθ hatası medyan 0.50°.

## Üçgen denetimi (F5-2): 2339 ardışık çift, iki kare de uyduyla iyi eşleşmiş

| bölge | görüntü ↔ uydu | görüntü ↔ GPS | uydu ↔ GPS |
|---|---|---|---|
| 01 | 16.2 m | 23.0 m | 24.3 m |
| 04 | 5.8 m | 27.8 m | 31.2 m |
| 05 | 16.5 m | 25.6 m | 36.9 m |
| 08 | 64.2 m | 35.7 m | 56.7 m |
| 09 | 8.1 m | 11.3 m | 15.9 m |
| 10 | 373.7 m | 15.0 m | 378.0 m |
| 11 | 8.9 m | 37.0 m | 37.7 m |
| hepsi | 13.1 m | 25.6 m | 31.6 m |
