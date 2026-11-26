# Faz 2A — etiket düzeltmesi (karar F2-4)

Bölge × uçuş yönü başına c = güvenilir karelerin medyan d'si (n ≥ 20; değilse c = 0). φ_corr = φ + c. Yalnızca train/val; calib değişmez. `data/m1/labels_v2.csv` sha256 `9d1eb336a500cc5e…`.

| bölge | yön | kare | medyan φ | c | c'nin dayandığı kare |
|---|---|---|---|---|---|
| 01 | 0 | 400 | 167° | -4.5° | 140 |
| 01 | 1 | 417 | 30° | -13.5° | 190 |
| 04 | 0 | 369 | 351° | +1.1° | 95 |
| 04 | 1 | 369 | 169° | -4.3° | 66 |
| 08 | 0 | 528 | 111° | -24.0° | 67 |
| 08 | 1 | 505 | 278° | +15.6° | 73 |
| 09 | 0 | 384 | 312° | +2.6° | 98 |
| 09 | 1 | 382 | 127° | -2.1° | 66 |
| 10 | 0 | 72 | 169° | 0 (ham; n az) | 0 |
| 10 | 1 | 72 | 326° | 0 (ham; n az) | 0 |

**Birini-dışarıda-bırak (bacağın düzeltmesi yalnızca aynı yöndeki diğer bacaklardan):**

| bölge | sınanan bacak | güvenilir kare | medyan \|d\| önce | sonra | \|d\| > 10° önce | sonra |
|---|---|---|---|---|---|---|
| 01 | 10 | 330 | 9.5° | 2.3° | %48 | %0 |
| 04 | 8 | 161 | 2.2° | 1.0° | %0 | %0 |
| 08 | 9 | 140 | 18.0° | 1.7° | %92 | %4 |
| 09 | 6 | 164 | 2.4° | 0.6° | %1 | %1 |
| hepsi | 33 | 795 | 5.2° | 1.4° | %36 | %1 |
