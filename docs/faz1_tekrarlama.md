# Faz 1 — Tekrarlama (1D) ve taban çizgileri (1F)

> Donanım: RTX 5050 Laptop 8 GB, Windows 11, torch 2.11.0+cu128, romav2 2.0.1.
> Her sayı `results/runs.csv`'de commit, config özeti ve tohumla kayıtlı.

## 1D. RoMa v2 tekrarlama

| Kıyas | Betik | Ayar | Rapor | Bizim | Fark | Durum |
|---|---|---|---|---|---|---|
| WxBS v1.1 (32 çift) | `21_repro_wxbs.py` | wxbs | mAA@10px 55,4 | **55,91 ± 0,00** (5 tohum) | +0,51 | ✅ ±1 içinde |
| MegaDepth-1500 | `20_repro_mega1500.py` | mega1500 | AUC@5/10/20 62,8 / 77,0 / 86,6 | – | – | ⏳ sunucuda |
| MegaDepth-C4 | `20_repro_mega1500.py --c4` | mega1500 | AUC@5 60,0 → 53,45 | – | – | ⏳ sunucuda |

**WxBS notları**
- mAA@10px = **0–10 px eşiklerinin (11 eşik) PCK ortalaması**. 0 hariç (1–10 px) tanım 61,50 veriyor ve
  rapordaki sayıyla uyuşmuyor; makaledeki tanım 0'ı içeriyor.
- WxBS değerlendirmesi yoğun warp haritasını doğrudan okur (örnekleme yok) → tohumlar arası fark 0.
- v1.1 test alt kümesi **32 çift** (MIMARI'de 34 yazıyordu).
- romav2 2.0.1'in benchmark'ındaki görselleştirme (`vis.py`) çift yönlü ayarda tensör boyutu
  uyuşmazlığıyla çöküyor; ölçümü etkilemediği için betikte devre dışı bırakıldı.
- Süre: 32 çift ≈ 2 dk.

**MegaDepth-1500 / C4 (sunucu için hazır)**
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

### Hız ve bellek (`23_speed.py`, gerçek drone↔uydu çifti, 1400 m karo)

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

Önbellek: `cache/dev_base` (ayar `base`; dizüstünde üretildi — sunucunun mega1500 önbelleği gelince
tablo `--cache cache/v1` ile yeniden üretilecek). Komutlar:
```
python scripts/30_build_cache.py data/UAV_VisLoc_dataset --setting base --split calib --cache cache/dev_base
python scripts/30_build_cache.py data/UAV_VisLoc_dataset --setting base --split train --per-region 60 --cache cache/dev_base
python scripts/40_baselines.py feats --cache cache/dev_base --split train --ransac-px 3 5 8
python scripts/40_baselines.py feats --cache cache/dev_base --split calib --ransac-px 3 5 8
python scripts/40_baselines.py select --cache cache/dev_base       # → configs/baselines.yaml
python scripts/40_baselines.py eval --cache cache/dev_base --split calib
```

**Eşik seçimi (yalnızca eğitim, bölge başına 60 kare = 300 kare, 840 çift):** RANSAC eşiği {3, 5, 8} px
adaylarından satır 2'nin eğitim başarı@25'ini en yüksek yapan → **3 px** (≈ 4 m); inlier eşiği τ her satır
için eğitimde FAR-neg ≤ %5 veren en küçük değer → τ = {1: 567, 2: 1251, 3: 294} (`configs/baselines.yaml`).

**Sonuç (calib: 1.063 pozitif, 1.931 negatif çift; GA %95, 2000 tekrar, 50 karelik uçuş blokları):**

| Satır | Yöntem | çağrı/kare | başarı@10 | başarı@25 [GA] | başarı@50 | başarı@25 (retsiz) | medyan hata (m, kabul) | medyan yön hatası | AUROC [GA] | FAR-neg (zor/kolay) | FAR-loc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | RoMa v2 (0°) + RANSAC | 1 | 0,044 | 0,198 [0,164, 0,232] | 0,406 | 0,202 | 29,4 | 4,2° | 0,950 [0,940, 0,960] | 0,042 (0,063/0,025) | 0,299 |
| 2 | 4 yön + en çok inlier | 4 | 0,070 | **0,291** [0,246, 0,339] | 0,537 | 0,305 | 26,6 | 4,1° | 0,935 [0,922, 0,947] | 0,026 (0,044/0,012) | 0,326 |
| 3 | 4 yön + en düşük entropi | 4 | 0,029 | 0,151 [0,116, 0,189] | 0,294 | 0,153 | 31,7 | 7,0° | **0,968** [0,958, 0,976] | 0,048 (0,069/0,030) | 0,276 |

McNemar (başarı@25, Holm): 2 > 1 (106'ya 7, p = 1,6e-23); 1 > 3 (62'ye 13, p = 8,4e-9); 2 > 3 (154'e 6,
p = 9,1e-38). Kırılımlar (yön, irtifa, arazi, bölge): `docs/sonuclar/faz1/table_dev_base_calib.md` (sonuç dosyaları `results/` git dışında olduğundan kopyası `docs/sonuclar/faz1/` altında).

**Yorum**
- Eşikler bölgeler arası taşınıyor: eğitimde FAR-neg = %5 için seçilen τ, calib'de %2,6–4,8 veriyor.
- 4 yön denemek başarı@25'i 0,20 → 0,29'a çıkarıyor (4 kat RoMa çağrısıyla) — H1'in (Modül 1: tek çağrıda
  aynı kazanç) motivasyonu. 0° satırının bile %20 başarması, RoMa v2'nin dönmeye kısmen dayanıklı olmasından.
- "En düşük entropi" konum seçiminde kötü (0,15) ama ret skorunda en iyi AUROC'u veriyor (0,968).
- Başarı oranları düşük; iki ana neden:
  1. **Alan farkı:** eşleşme ya tutuyor (hata 5–50 m, binlerce inlier) ya tamamen kaçıyor (yüzlerce m).
     Tarım/su ağırlıklı karelerde ve drone–uydu arasında mevsim farkı olan bölgelerde (08, 10, 02) kaçıyor.
  2. **GT kayması:** CSV `lat/lon` görüntü merkezi değil, drone'un GPS konumu (aşağıya bkz.) → doğru
     eşleşmelerde bile 15–40 m sistematik hata → başarı@10 çok düşük.
- Karo boyutu: eğitimden 30 karelik denemede 1000 m karo, 1400 m'den biraz daha iyi (başarı@25 0,20 vs 0,13;
  700 m: 0,17) — örneklem küçük, anlamlı değil. `tile_size_m` danışmanla belirlenecek; değişirse önbellek
  yeniden üretilmeli.

## Ekip kararı gerektiren bulgular

1. **GT konumu görüntü merkezi değil.** RoMa ile ölçülen görüntü merkezi ile CSV `lat/lon` arasındaki kayma,
   drone çerçevesinde kamera eğimiyle güçlü ilişkili: yanal kayma ↔ Omega (r = −0,88), ileri kayma ↔ Kappa
   (r = +0,58) + ≈ 2° sabit ileri kayma (README'deki "Omega = pitch, Kappa = roll" etiketleri ters görünüyor).
   Eğim modeli eğitim bölgelerinde medyan kaymayı 22 → 13 m'ye indiriyor ama calib'de iyileştirmiyor
   (28 → 28 m; 11'de eğim 23°'ye kadar). → Şimdilik GT = CSV (veri setinin standardı). Seçenekler: (a) böyle
   bırak, başarı@10'un tabanını raporla; (b) eğim düzeltmesini daha çok örnekle kur, `pairs_v2`.
2. `tile_size_m` (1400 m) ve `tile_offset_m` (±200 m) danışmanla kesinleşmeli (önbellek buna bağlı).
3. INTERFACES §1–2 değişiklikleri (Σ float32, ek çift alanları) onay bekliyor.

## Faz 1 kapanış kontrol listesi (MIMARI §14.4)

- [x] 1A veri doğrulama, checksum (`docs/faz1_veri_notlari.md`)
- [x] 1B coğrafi altyapı, yön tanımı ve ölçek (RoMa ile sayısal sağlama)
- [x] 1C `split.yaml` (dondurulmuş), `pairs_v1.csv` (birebir yeniden üretilebilir)
- [x] 0.4 hassasiyet birimi, 1E `matcher.py` + önbellek altyapısı + şema testi (`docs/faz1_matcher_onbellek.md`)
- [x] 1F metrikler, istatistik, taban çizgileri (calib tablosu, `dev_base` önbelleği ile)
- [x] 1D WxBS ±1 içinde; hız/bellek ölçüldü
- [ ] 1D MegaDepth-1500 + C4 (sunucu) → tamamlanınca `v0.0` etiketi
- [ ] Tam önbellek `cache/v1` (sunucu, `base` — K6) → taban çizgisi tablosunun `cache/v1` ile yenilenmesi
- [ ] Danışmana tek sayfalık özet
