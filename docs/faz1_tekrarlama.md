# Faz 1 — Tekrarlama (1D) ve taban çizgileri (1F)

> Donanım: WxBS ve ilk ölçümler RTX 5050 Laptop 8 GB (Windows 11); MegaDepth/C4, tam önbellek `cache/v1` ve
> taban çizgisi tablosu **üniversite sunucusunda (RTX 3090 24 GB, Ubuntu 22.04)**, 25–26.11.2026.
> torch 2.11.0+cu128, romav2 2.0.1.
> Her sayı `results/runs.csv`'de commit, config özeti ve tohumla kayıtlı.

## 1D. RoMa v2 tekrarlama

| Kıyas | Betik | Ayar | Rapor | Bizim | Fark | Durum |
|---|---|---|---|---|---|---|
| WxBS v1.1 (32 çift) | `21_repro_wxbs.py` | wxbs | mAA@10px 55,4 | **55,91 ± 0,00** (5 tohum) | +0,51 | ✅ ±1 içinde |
| MegaDepth-1500 (1500 çift) | `20_repro_mega1500.py` | mega1500 | AUC@5/10/20 62,8 / 77,0 / 86,6 | **62,98 ± 0,13 / 76,96 ± 0,11 / 86,55 ± 0,09** (5 tohum) | +0,18 / −0,04 / −0,05 | ✅ ±1 içinde |
| MegaDepth-C4 (1500 çift) | `20_repro_mega1500.py --c4` | mega1500 (800/1024 px) | AUC@5 60,02 → 53,45 (Ge vd., 2026; **576×576**) | **62,98 → 58,94 ± 0,08** (5 tohum) | AUC@5 +5,49 | ⚠️ sapma > 1 — farklı çözünürlük (aşağıya bkz.) |

**WxBS notları**
- mAA@10px = **0–10 px eşiklerinin (11 eşik) PCK ortalaması**. 0 hariç (1–10 px) tanım 61,50 veriyor ve
  rapordaki sayıyla uyuşmuyor; makaledeki tanım 0'ı içeriyor.
- WxBS değerlendirmesi yoğun warp haritasını doğrudan okur (örnekleme yok) → tohumlar arası fark 0.
- v1.1 test alt kümesi **32 çift** (MIMARI'de 34 yazıyordu).
- romav2 2.0.1'in benchmark'ındaki görselleştirme (`vis.py`) çift yönlü ayarda tensör boyutu
  uyuşmazlığıyla çöküyor; ölçümü etkilemediği için betikte devre dışı bırakıldı.
- Süre: 32 çift ≈ 2 dk.

**MegaDepth-1500 / C4 — sonuç (sunucu, 25.11.2026, 5 tohum × 1500 çift, ≈ 18–20 dk/tohum)**
- MegaDepth-1500: tohumlar 62,75–63,12 (AUC@5); makaleyle ±0,2 puan içinde. Duman testi (25 çift) bu makinede
  romav2'nin kendi `Mega1500.benchmark`'ıyla **birebir aynı** (55,61 / 72,85 / 84,33; Kaggle T4'te aynı alt küme
  57,63 / 73,19 / 84,11 — fark donanımdan, iki makinede de kendi orijinaliyle birebir).
- C4, yön kırılımı (AUC@5, 5 tohum ortalaması): **0° 63,19 · 90° 60,05 · 180° 54,04 · 270° 58,55**; toplam 58,94
  (AUC@10/20 = 73,48 / 83,78). 0° çiftler MegaDepth-1500 ile tutarlı → döndürme kodu doğru. Dönme başarıyı
  4,0 puan düşürüyor (raporda 6,55).
- **Sapma:** hedef 53,45 bizim sonucumuzdan 5,5 puan düşük. Hedefle birlikte verilen döndürmesiz değer (60,0) de
  RoMa v2'nin MegaDepth-1500 sonucundan (62,8) farklı → bu iki sayı büyük olasılıkla başka bir model/ayardan ya
  da başka bir C4 tanımından (ör. her iki görüntünün döndürülmesi, farklı dönme dağılımı) geliyor. Kaynağı
  henüz doğrulanmadı. MIMARI 1D B planı: sapma nedeniyle raporlanır, karşılaştırmalar kendi tekrarımıza göre
  (döndürmesiz 62,98, C4 58,94). Sonuç dosyaları: `docs/sonuclar/faz1/repro_mega1500.json`, `repro_c4.json`.
- **Hedefin kaynağı (26.11):** 2209 önerisindeki atıf REDI-Match (Ge vd., 2026, arXiv 2606.24330). Sayılar aynı
  makalenin iki tablosundan: Tablo 3 düz MegaDepth RoMa v2 **60,02** / 74,77 / 85,19; Tablo 1 MegaDepth-C4 RoMa v2
  **53,45** / 68,93 / 80,09 (RoMa v1: 59,56 → 25,90). Makale **bütün yöntemleri sabit 576×576 çözünürlükte**
  değerlendiriyor; RoMa v2'yi hangi ayarla koştuğunu, kendi deposunda RoMa v2 kodu olmadığı için de yeniden
  üretilebilirliğini vermiyor. C4 çiftleri `rotation_gt.json` ile hazır veriliyor (B görüntüsü k·90° döndürülmüş);
  açının nasıl seçildiği yazılmıyor. Önerideki "C4 testinde 60,02'den 53,45'e" ifadesi düz → C4 karşılaştırması olarak
  okunmalı.

  | | düz MegaDepth AUC@5 | C4 AUC@5 | dönme kaybı |
  |---|---|---|---|
  | Edstedt vd. (2025), RoMa v2 makalesi, mega1500 (800/1024) | 62,8 | – | – |
  | **Bizim, mega1500 (800/1024), 5 tohum** | **62,98** | **58,94** | −4,04 |
  | Ge vd. (2026), 576×576 | 60,02 | 53,45 | −6,57 |

  **Sonuç:** fark protokolden değil (bizim C4 tanımımız Steerers'la aynı: ikinci görüntü (i mod 4)·90°), büyük
  olasılıkla **çözünürlükten**: Ge vd. düz MegaDepth'te de RoMa v2'nin kendi sonucundan 2,8 puan düşük; düşük
  çözünürlük döndürülmüş çiftlerde daha çok kayıp verebilir. Bu, ölçülerek doğrulanmadı. Karşılaştırmalar kendi
  tekrarımıza (62,98 → 58,94) göre yapılır (MIMARI 1D B planı). Steerers (arXiv 2312.02152) C4 tanımı; RoMa v2
  makalesinde (arXiv 2511.15706) döndürülmüş MegaDepth sonucu yok.

**MegaDepth-1500 / C4 — yöntem notları**
- **Kaggle T4 duman testi (24.11.2026, `--limit 5` = 25 çift, tohum 0):** AUC@5/10/20 = 57,63 / 73,19 / 84,11;
  romav2'nin kendi `Mega1500.benchmark`'ı aynı alt kümede **birebir aynı** (57,626 / 73,188 / 84,111).
  Alt küme sayıları hedefle (tam 1500 çift) karşılaştırılamaz. Hız: 7,0 s/çift → tam tohum ≈ 2,9 sa.
  Ortam: Tesla T4 (sm_75, bf16 öykünmeli), Python 3.13, torch 2.11.0+cu130 (Kaggle'ın hazır sürümü), numpy 2.5.4.
- romav2 2.0.1 `pose_auc`, NumPy 2.5'te kaldırılan `np.trapz`'ı çağırıyor → `guvenloc/repro.py` uyum yaması
  (`np.trapz = np.trapezoid`, aynı fonksiyon; `tests/test_repro.py`).
- `20_repro_mega1500.py`, romav2 `Mega1500.benchmark` döngüsünün birebir kopyasıdır; `--check-original`
  aynı alt kümede orijinali de koşup sonucun aynı olduğunu doğrular (C4'ün 0° yolu = orijinal).
- C4: i. çiftin ikinci görüntüsü `np.rot90(·, i mod 4)`; K ve poz aynı dönmeyle güncellenir
  (`guvenloc/repro.py::rot90_camera`, `tests/test_repro.py`: döndürülmüş kamerayla izdüşüm,
  döndürülmüş görüntüdeki noktalarla 1e-8 içinde örtüşüyor). Sonuç ayrıca yön bazında (0/90/180/270°) verilir.
- Komutlar (veri: romav2 `scripts/eval_prep.sh`'in MegaDepth satırları → `data/megadepth/`):
  ```
  python scripts/20_repro_mega1500.py --data data/megadepth --limit 5 --check-original   # duman testi
  python scripts/20_repro_mega1500.py --data data/megadepth --seeds 0 1 2 3 4
  python scripts/20_repro_mega1500.py --data data/megadepth --c4 --seeds 0 1 2 3 4
  ```
- Bu dizüstünde de mümkün: mega1500 çıkarımı 4,4 s/çift, tek tohumlu tam koşu ≈ 4 saat
  (romav2'nin `sample()`'ı tepe 7,9 GB — 8 GB kartta sınırda).

### Hız ve bellek (`23_speed.py`, gerçek drone↔uydu çifti)

**Sunucu (RTX 3090, 25.11, uyarlamalı karo):** base: RoMa 0,16 s, `Matcher.match` 0,26 s, tepe 3,6 GB;
mega1500: RoMa 0,54 s, `Matcher.match` 0,66 s, tepe 7,3 GB. Önbellek betiği (G/Ç dahil) base 0,25 s/dosya →
tam `cache/v1` (86.620 dosya) 6 sa (`docs/sonuclar/faz1/speed_3090.json`).

**Dizüstü (RTX 5050 Laptop 8 GB, 1400 m karo):**

| Ayar | Izgara | RoMa çıkarımı | `Matcher.match` | Tepe bellek (Matcher) | romav2 `sample()` | bizim örnekleyici | romav2 `sample()` tepe |
|---|---|---|---|---|---|---|---|
| turbo | 320 | 0,13 s | 0,33 s | 2,1 GB | 0,37 s | 0,15 s | 7,5 GB |
| fast | 512 | 0,25 s | 0,46 s | 2,7 GB | 0,68 s | 0,15 s | 7,5 GB |
| base | 640 | 0,39 s | 0,62 s | 3,6 GB | 0,82 s | 0,15 s | 7,6 GB |
| mega1500 | 1024, çift yönlü | 4,39 s | 4,37 s | 7,3 GB | 0,82 s | 0,20 s | 7,9 GB |

- romav2'nin `sample()`'ı KDE için 40k×40k mesafe matrisini tek seferde kurduğundan tek başına 7,5–7,9 GB
  tepe yapıyor; Gün 1'deki "mega1500 7,1 GB" ve 8 GB kartta 45–56 s/çift gözlemi bundandı.
  `matcher.balanced_sample` aynı algoritmayı parça parça yapar (aynı tohumla birebir aynı eşleşmeler).
- Önbellek hızı (önbellek betiği, G/Ç dahil): base ≈ 0,75 s/dosya. Tam önbellek (≈ 79 bin dosya):
  base ≈ 17 sa, mega1500 ≈ 4 gün → mega1500 önbelleği sunucuda.

## 1F. Taban çizgileri — kalibrasyon bölgeleri (05 Yunnan, 11 Shandan)

Önbellek: **`cache/v1`** (sunucu, ayar `base` — K6; uyarlamalı karo — K2; 86.620 dosya, bozuk 0). Komutlar
(`scripts/faz1_3090.py` `baselines` adımı):
```
python scripts/40_baselines.py feats --cache cache/v1 --split train --ransac-px 3 5 8
python scripts/40_baselines.py feats --cache cache/v1 --split calib --ransac-px 3 5 8
python scripts/40_baselines.py select --cache cache/v1       # → configs/baselines.yaml
python scripts/40_baselines.py eval --cache cache/v1 --split calib
```

**Eşik seçimi (yalnızca eğitim, 5 bölgenin tamamı, 10.350 çift):** RANSAC eşiği {3, 5, 8} px adaylarından satır 2'nin
eğitim başarı@25'ini en yüksek yapan → **5 px**; inlier eşiği τ her satır için eğitimde FAR-neg ≤ %5 veren en küçük
değer → τ = {1: 653, 2: 1462, 3: 380} (`configs/baselines.yaml`).

**Sonuç (calib: 1.063 pozitif, 2.035 negatif çift; GA %95, 2000 tekrar, 50 karelik uçuş blokları):**

| Satır | Yöntem | çağrı/kare | başarı@10 | başarı@25 [GA] | başarı@50 | başarı@25 (retsiz) | medyan hata (m, kabul) | medyan yön hatası | AUROC [GA] | FAR-neg (zor/kolay) | FAR-loc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | RoMa v2 (0°) + RANSAC | 1 | 0,043 | 0,214 [0,176, 0,252] | 0,412 | 0,215 | 27,9 | 4,1° | 0,953 [0,943, 0,963] | 0,068 (0,092/0,046) | 0,285 |
| 2 | 4 yön + en çok inlier | 4 | 0,069 | **0,319** [0,280, 0,361] | 0,575 | 0,335 | 25,1 | 3,8° | 0,927 [0,914, 0,940] | 0,055 (0,084/0,028) | 0,325 |
| 3 | 4 yön + en düşük entropi | 4 | 0,030 | 0,163 [0,124, 0,204] | 0,313 | 0,163 | 30,5 | 5,4° | **0,963** [0,954, 0,972] | 0,063 (0,082/0,045) | 0,260 |

McNemar (başarı@25, Holm): 2 > 1 (120'ye 8, p = 1,8e-26); 1 > 3 (64'e 10, p = 9,0e-11); 2 > 3 (175'e 9,
p = 1,4e-40). Kırılımlar (yön, irtifa, arazi, bölge): `docs/sonuclar/faz1/table_v1_calib.md`.

**Önceki tabloyla karşılaştırma** (dizüstü `dev_base` önbelleği, sabit 1400 m karo, eğitimde 300 kare;
`docs/sonuclar/faz1/table_dev_base_calib.md`): başarı@25 satır 1/2/3 = 0,198 / 0,291 / 0,151 → **0,214 / 0,319 / 0,163**.
Üç satırda ve iki calib bölgesinde de artış — K2'de (uyarlamalı karo) eğitimde görülen kazanç calib'de de tutuyor.
Satır sıralaması (2 > 1 > 3) ve AUROC'lar değişmedi.

**Eğitim bölgelerinde satır 2 (K5 için; test bölgelerine dokunulmadı):**

| küme | bölge | arazi | drone tarihi | pozitif | başarı@25 | kabul oranı | > 100 m kaçan | FAR-neg |
|---|---|---|---|---|---|---|---|---|
| train | 01 | kent | 2018-09-17 | 817 | 0,404 | 0,590 | 0,370 | 0,107 |
| train | 04 | kent | 2018-10-24 | 738 | 0,218 | 0,618 | 0,278 | 0,007 |
| train | 08 | kent (su) | 2019-06-14 | 1033 | 0,136 | 0,221 | 0,663 | 0,060 |
| train | 09 | kent | 2019-06-15 | 766 | 0,214 | 0,295 | 0,550 | 0,012 |
| train | 10 | tarım | 2018-09-05 | 144 | 0,049 | 0,160 | 0,750 | 0,111 |
| calib | 05 | tarım | 2016-06-27 | 473 | 0,290 | 0,577 | 0,347 | 0,104 |
| calib | 11 | tarım | 2023-10-19 | 590 | 0,342 | 0,698 | 0,220 | 0,019 |

Uydu görüntülerinin çekim tarihi veri setinde yok (tif etiketlerinde ve `satellite_coordinates_range.csv`'de tarih
alanı bulunmuyor); bu yüzden drone–uydu tarih farkı hesaplanamadı, yalnızca drone tarihleri verilmiştir.
Uydu tarihleri veri seti yazarlarından/makaleden öğrenilirse tablo tamamlanır.

**Yorum**
- **Eşikler bölgeler arası tam taşınmıyor:** eğitimde FAR-neg = %5 için seçilen τ, calib'de %5,5–6,8 veriyor (eski
  küçük örneklemde %2,6–4,8 idi). Bölge başına FAR-neg %0,7 (04) ile %11 (01, 10; calib'de 05) arasında değişiyor.
  Sabit eşikle yanlış kabul hedefi bölgeden bölgeye tutmuyor → Modül 3'ün (conformal kalibrasyon) çözmesi gereken
  sorunun doğrudan kanıtı (H3).
- 4 yön denemek başarı@25'i 0,21 → 0,32'ye çıkarıyor (4 kat RoMa çağrısıyla) — H1'in (Modül 1: tek çağrıda
  aynı kazanç) motivasyonu. 0° satırının bile %21 başarması, RoMa v2'nin dönmeye kısmen dayanıklı olmasından.
- "En düşük entropi" konum seçiminde kötü (0,16) ama ret skorunda en iyi AUROC'u veriyor (0,963).

**Zor durumlar (K5).** Başarının düşük olması bir hata değil; iki ayrı mekanizmadan geliyor:
1. **Eşleşme ya tutuyor ya tamamen kaçıyor.** Tutan karelerde hata 5–50 m ve binlerce inlier var; kaçanlarda hata
   yüzlerce metre (08, 09 ve 10'da medyan hata 450–570 m, karelerin %55–75'i 100 m'nin dışında). Kaçmanın
   yoğunlaştığı yerler: geniş su yüzeyleri ve kanallar (08, 09 — Huzhou-3), dokusuz tarla ve sulak alan (10 —
   Huailai). Bu bölgelerde kabul oranı da düşük (%16–30), yani taban çizgisi çoğu kaçışı doğru reddediyor; sorun
   başarının değil kapsamanın düşüklüğü. Kent dokusu belirgin bölgelerde (01, 04) ve calib'de (05, 11) başarı
   0,22–0,40. Drone tarihleri yaz (Haziran: 05, 08, 09) ile sonbahar (Eylül–Ekim: 01, 04, 10, 11) arasında; uydu
   tarihleri bilinmediğinden mevsim etkisi ayrıştırılamıyor.
2. **GT kayması (K1).** CSV `lat/lon` görüntü merkezi değil, drone'un GPS konumu: doğru eşleşmelerde bile medyan
   ≈ 22–25 m sistematik kayma var (kamera eğimiyle ilişkili; aşağıya bkz.). Bu, başarı@10'u (0,04–0,07) doğrudan
   sınırlıyor; başarı@25 ve @50 daha az etkileniyor.
Düşük taban, M2–M4'ün katkısını göstermek için iyi bir başlangıç: M1 yön çağrılarını azaltır, M2 konumu inceltir,
M3 eşiği bölgeden bağımsız hâle getirir.

## Belirsizlik ölçeği `sigma_scale` (§4c, `scripts/42_sigma_scale.py`)

Eğitim pozitiflerinde satır 2'nin kabul ettiği ve hatası < 25 m olan 802 çiftin RANSAC inlier'ları (5,49 milyon
eşleşme): artık r = v − H(u) (karo pikseli), m = rᵀ Σ_B⁻¹ r; Gauss NLL'yi en küçükleyen ölçek s² = mean(m)/2.

| küme | inlier | s (NLL) | %95 elips kapsaması (s ile) | %68 kapsama | %95 kapsama (ham Σ) |
|---|---|---|---|---|---|
| train | 5,49 M | **0,927** (seçilen) | %93,2 | %73,9 | %94,9 |
| calib | 2,51 M | (0,816) | %95,7 | %84,4 | %96,5 |

- **s ≈ 0,93:** RoMa v2'nin Σ'sı (Faz 0.4'teki birim düzeltmesiyle) gerçek artıklarla aynı mertebede; yalnızca
  %7 küçültme gerekiyor. 0.4'ün birim çözümünün gerçek veride de doğru olduğunun bağımsız sağlaması.
- Kapsama %95'e yakın ama dağılım Gauss değil: %68 elipsi eğitimde %74, calib'de %84 kapsıyor (merkez dar,
  kuyruk ağır). Bölgeler arasında s 0,69 (10) – 1,02 (04) arasında değişiyor. M2 tek ölçek kullanır;
  bölge farkı M3'ün (conformal) kalibrasyonuna kalır.
- Sınırlama: yalnızca RANSAC inlier'ları (|r| < 5 px) kullanıldığından kuyruk kesik; s hafifçe küçük kestirilmiş
  olabilir (kapsama eşleştiren ölçek eğitimde 1,01). Kural önceden konduğu gibi uygulandı.
- `configs/base.yaml: romav2.sigma_scale = 0.9272`; önbellek ham Σ saklar, tüketici kuralı INTERFACES §1'de.
  Ayrıntı: `docs/sonuclar/faz1/sigma_scale.md`.

## Kararlar (bkz. faz1_durum.md §3)

1. **GT konumu görüntü merkezi değil — K1: konu kapandı, GT = CSV.** RoMa ile ölçülen görüntü merkezi ile CSV
   `lat/lon` arasındaki kayma drone çerçevesinde kamera eğimiyle güçlü ilişkili (calib'de r(yanal, Omega) = −0,79/−0,90,
   r(ileri, Kappa) = 0,93/0,88). Önceden konan kuralla (`scripts/41_gt_tilt.py`; eğitimde 1.358 kare, satır 2 kabul +
   hata < 100 m) kurulan doğrusal model `yanal = 1,8 − 7,0·Ω + 1,0·κ`, `ileri = 6,1 − 1,2·Ω + 5,7·κ` (m, derece)
   calib'de medyan kaymayı **24,9 → 19,3 m (%22,3, GA [%14, %28])** azaltıyor; eşik %30'un altında →
   **`pairs_v2` üretilmez, GT = CSV.** Bölgeler arasında katsayılar farklı (04'te %41, 01'de %3 azalma) — kamera
   montajı/zamanlama farkı olabilir; tek doğrusal model genellenmiyor. Ek bilgi (karar için kullanılmadı):
   AGL'ye bölünmüş model calib'de yalnızca %6,5 azaltıyor. Ayrıntı: `docs/sonuclar/faz1/k1_tilt.md`.
2. `tile_size_m` / `tile_offset_m` — K2 ile karara bağlandı (uyarlamalı karo, ±200 m).
3. INTERFACES §1–2 değişiklikleri (Σ float32, ek çift alanları) — onaylandı (25.11, faz1_durum K3).

## Faz 1 kapanış kontrol listesi (MIMARI §14.4)

- [x] 1A veri doğrulama, checksum (`docs/faz1_veri_notlari.md`)
- [x] 1B coğrafi altyapı, yön tanımı ve ölçek (RoMa ile sayısal sağlama)
- [x] 1C `split.yaml` (dondurulmuş), `pairs_v1.csv` (birebir yeniden üretilebilir)
- [x] 0.4 hassasiyet birimi, 1E `matcher.py` + önbellek altyapısı + şema testi (`docs/faz1_matcher_onbellek.md`)
- [x] 1F metrikler, istatistik, taban çizgileri (calib tablosu)
- [x] 1D WxBS ±1 içinde; hız/bellek ölçüldü
- [x] 1D MegaDepth-1500 ±1 içinde (62,98 / 76,96 / 86,55); C4 sapması nedeniyle raporlandı
- [x] Tam önbellek `cache/v1` (sunucu, `base` — K6; 86.620 dosya, bozuk 0) → taban çizgisi tablosu `cache/v1` ile yenilendi
- [x] K1 eğim analizi (kural: azalma %22,3 < %30 → GT = CSV)
- [x] `sigma_scale` kalibrasyonu: s = 0,9272 (eğitim, NLL); %95 kapsama eğitim %93,2, calib %95,7
- [x] C4 hedefinin kaynağı: REDI-Match (Ge vd., 2026), 576×576; sapma çözünürlük farkıyla açıklandı (ölçülmedi)
- [ ] PR #2 → `main`, `v0.0` etiketi
- [ ] Danışmana tek sayfalık özet
