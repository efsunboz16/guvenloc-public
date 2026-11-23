# Faz 1A — UAV-VisLoc veri notları

Doğrulama: `python scripts/11_verify_data.py data/uav_visloc_sample --expect-frames 768 --expect-maps 1 --checksums`
Checksum listesi: `docs/checksums/uav_visloc_sample.sha256`

## Örnek paket (`UAV_VisLoc_example.zip`, 2,19 GB)
- Yalnızca bölge **03** (Taizhou-1): 768 drone karesi, 768 CSV satırı, eksik değer yok.
- Uydu haritası `satellite03.tif`: 35092 × 24308 px, 3 bant uint8, **EPSG:4326**,
  sınırlar koordinat CSV'siyle birebir aynı.
- Koordinat dosyasının adı pakette **boşluklu**: `satellite_ coordinates_range.csv`
  (betik esnek arıyor). İçindeki `mapname` sütunu `03.tif` yazıyor, dosya ise `satellite03.tif`.

## CSV biçimleri (paket içi README_example.txt'den, doğrulandı)

`satellite_coordinates_range.csv`:
`mapname, LT_lat_map, LT_lon_map, RB_lat_map, RB_lon_map, region`

`NN/NN.csv`:
`num, filename, date, lat, lon, height, Omega, Kappa, Phi1, Phi2`
- `lat`, `lon`: drone karesinin **merkezinin** koordinatı.
- `height`: çekim yüksekliği (03'te 465–468 m; birim belirtilmemiş, metre varsayılıyor).
- `Omega` = pitch, `Kappa` = roll (03'te ±5° içinde → kareler yaklaşık nadir).
- `Phi1`, `Phi2` = yaw. **Phi1 daha güvenilir** (README). 03'te Phi1 ∈ [−44°, 128°].
  İşaret ve referans (kuzeye göre, saat yönü mü?) Faz 1B bindirme sağlamasıyla kesinleşecek.
- Raporda "yön (Phi) 2 bölgede yok" deniyor → tam pakette hangi bölgelerde Phi1 boş,
  `11_verify_data.py` eksik değer oranıyla gösterecek.

## Faz 1B'yi etkileyen bulgular
1. **Uydu pikseli metrede kare değil.** EPSG:4326 ızgarası derece cinsinden eşit adımlı;
   32,3° enlemde 1 px ≈ **0,253 m (doğu-batı) × 0,297 m (kuzey-güney)**.
   Rapordaki "0,3 m" yalnızca kuzey-güney yönü için doğru.
   → `geo.py` metre hesabını iki eksende ayrı yapmalı; RoMa v2'ye verilecek karolar
   yerel metrik ızgaraya (ör. UTM veya yerel ENU, kare piksel) yeniden örneklenmeli.
   Aksi hâlde drone↔uydu ölçeği eksenlere göre %17 farklı olur.
2. **Drone JPG'lerinde EXIF yok** (kamera modeli, odak uzaklığı yok). Kareler 3976 × 2652 px.
   → Ölçek `s`, kamera iç parametrelerinden hesaplanamaz. MIMARI.md §1B'deki yedek yol
   uygulanacak: eğitim bölgelerinde RoMa v2 homografisinden ölçülen ölçek,
   `height`'a karşı regresyonla bağlanır (`GSD_drone = k · height`).

## Tam paket (`UAV_VisLoc_dataset`, 16,4 GB) — 1A doğrulaması

Doğrulama: `python scripts/11_verify_data.py data/UAV_VisLoc_dataset --expect-frames 6774 --expect-maps 11 --checksums`
Checksum listesi: `docs/checksums/UAV_VisLoc_dataset.sha256` (6801 dosya; türetilmiş `*_mosaic.vrt` hariç).
Paket kökü `UAV_VisLoc_dataset/` adıyla geliyor; betikler kök yolunu argüman olarak alır.

| Bölge | Ad | Kare | Harita (px) | İrtifa `height` (m) | Not |
|---|---|---|---|---|---|
| 01 | Changjiang-20 | 817 | 9774 × 26762 | 403–409 | |
| 02 | Changjiang-23 | 1071 | 11482 × 34291 | 402–408 | |
| 03 | Taizhou-1 | 768 | 35092 × 24308 | 465–468 | |
| 04 | Taizhou-6 | 738 | 18093 × 38408 | 539–547 | |
| 05 | Yunnan | 473 | 9394 × 6144 | 2307–2319 | `height` büyük olasılıkla deniz seviyesine göre |
| 06 | Zhuxi | 344 | 8082 × 9780 | 499–842 | 6 kare harita dışında |
| 07 | Donghuayuan | 30 | 3000 × 170 | 689–690 | **Yön/poz sütunu yok**; harita ≈ 50 m yüksekliğinde; 2 kare harita dışında |
| 08 | Huzhou-3 | 1033 | 43421 × 16294 | 545–553 | İlk kareler su üstünde |
| 09 | Huzhou-3 | 766 | 44800 × 33280 (2×2 parça) | 546–547 | Koordinat CSV'si tif'ten 52 m'ye kadar sapıyor |
| 10 | Huailai | 144 | 6593 × 5077 | 772–774 | |
| 11 | Shandan | 590 | 29592 × 16582 | 1971–2575 | `height` deniz seviyesine göre görünüyor; pitch 23°'ye kadar |

Bulgular:
1. **Toplam kare 6.774**, rapor/makaledeki 6.742 değil (+32). Her bölgede kare sayısı = CSV satırı = CSV'deki
   dosya adları; eksik/fazla dosya yok. Fark makale ile paket sürümü arasında.
2. **Yön (Phi1/Phi2) yalnızca 07'de yok** (sütunlar hiç yok). Yunnan (05) ve Shandan (11)'da Phi1 tam dolu.
   Raporda "Phi iki bölgede yok — Yunnan, Shandan" deniyordu; bu, tam pakette doğru değil.
   → `configs/split.yaml` için kalibrasyon bölgesi seçiminin gerekçesi ("yön etiketi yok") geçersiz;
   ayrım yeniden konuşulmalı (1C).
3. **09 dört parça** (`satellite09_RR-CC.tif`, aynı CRS ve çözünürlük, bitişik). `io_visloc.satellite_source`
   bölge klasörüne `satellite09_mosaic.vrt` yazar ve bölge tek harita olarak okunur.
4. **Georeferans tif'ten okunur.** 09 dışında CSV ile tif sınırları birebir aynı; 09'da CSV
   [sol, üst, sağ, alt] = [+30,7, −52,6, −25,7, +1,4] m kayık. CSV kullanılsaydı 09'daki GT konumları kayardı.
5. **08 ve 09 aynı bölge adını taşıyor (Huzhou-3)** ve haritaları üst üste biniyor
   (08: 120,136–120,253° D; 09: 120,030–120,150° D). Bölge ayrık protokolde 08 ve 09 aynı kümede olmalı.
6. **07 pratikte kullanılamaz**: 30 kare, yön yok, harita bandı ≈ 50 m. Ayrımda dışarıda bırakılması önerilir.

## 1B — Yön (Phi) tanımı ve ölçek: RoMa v2 ile sayısal sağlama

Betik: `python scripts/13_roma_heading_scale.py data/UAV_VisLoc_dataset` (ayar `base`, bölge başına 12 kare,
1000 m kuzey-yukarı karo, drone 4 yönde denenir, RANSAC benzerlik dönüşümü). Süre ≈ 7 dk (RTX 5050 8 GB).
Ham sonuç: `results/faz1/roma_heading_scale.csv`. Özet (yalnızca ≥ 500 inlier'lı kareler):

| Bölge | Güvenilir / 12 | bearing − Phi1 (dairesel ort.) | yayılım (medyan) | m/px (tam çöz.) | ayak izi genişliği | ayak izi / height | merkez kayması (medyan) |
|---|---|---|---|---|---|---|---|
| 01 | 7 | −7,0° | 4,1° | 0,10 | 383 m | 0,95 | 17 m |
| 02 | 3 | (güvenilmez) | 28° | 0,09 | 362 m | 0,89 | 53 m |
| 03 | 10 | −4,0° | 2,7° | 0,11 | 457 m | 0,98 | 15 m |
| 04 | 9 | −1,6° | 3,1° | 0,13 | 535 m | 0,98 | 41 m |
| 05 | 5 | −0,8° | 2,9° | 0,12 | 374 m | 0,16 | 27 m |
| 06 | 11 | +0,4° | 5,2° | 0,08 | 307 m | 0,37 | 22 m |
| 07 | 3 | – (yön yok) | – | 0,01 | 27 m | – | 8 m |
| 08 | 4 | −4,7° | 20° | 0,14 | 540 m | 0,98 | 14 m |
| 09 | 3 | +0,6° | 2,2° | 0,14 | 560 m | 1,03 | 12 m |
| 10 | 2 | (az örnek) | 10° | 0,10 | 286 m | 0,37 | 28 m |
| 11 | 11 | −2,0° | 1,7° | 0,19 | 772 m | 0,30 | 30 m |

Sonuçlar (geo.py docstring'ine işlendi):
1. **Phi1 = görüntü üst kenarının pusula yönü** (kuzey 0, saat yönünde pozitif). İşaret ters değil,
   180° kayma yok. Kuzey-yukarı kare için görüntü saat yönünde Phi1 döndürülür
   (`geo.north_up_rotation_cw_deg`). İlk bindirme görsellerinde 01/02 için "+Phi+180" gibi görünen
   eşleşme göz yanılgısıydı; RoMa ölçümü 01'de de +Phi'yi veriyor.
2. Fark çoğu bölgede birkaç derece, ama **uçuş bacağına bağlı**: 01'de kuzeydoğuya giden bacaklarda
   −10…−14°, güneye gidenlerde −3°; 08'de bacaklara göre −26° / +13…+19°. Muhtemelen rüzgâr kaynaklı
   sapma (crab) veya gimbal. → M1 etiket gürültüsü (MIMARI 2A) için önemli; etiket olarak Phi1 mi
   yoksa RoMa ölçümü mü kullanılacağı Faz 2'de karar verilecek.
3. **`height` her bölgede yere göre değil.** 01/03/04/08/09'da ayak izi ≈ 1,0 × height (yatay görüş
   açısı ≈ 53°) → yere göre. 05/06/10/11'de oran 0,16–0,37 → deniz seviyesine göre. → Ölçek `s`
   bölge başına RoMa ölçümünden (veya bölge başına height ofsetiyle) bağlanmalı; `configs/pairs.yaml`.
4. GT konumları RoMa ile 10–40 m içinde tutarlı (karo merkezi kayması). 06'da üç karede ≈ 350 m kayma
   var (tekrarlayan desen ya da yanlış eşleşme; incelenecek).
5. 02 (tarım arazisi) ve 10'da eşleşme zayıf: görüntüler arasında mevsim farkı olabilir. Zor bölge adayı.

## 1C — Bölge ayrımı ve çift üretimi

`configs/split.yaml` (dondurulmuş, v1): train 01, 04, 08, 09, 10 (3.498 kare) | calib 05, 11 (1.063) |
test 02, 03, 06 (2.177). 07 dışarıda; 06_0093–0098 (bozuk GPS, harita dışı) dışarıda; 08+09 tek grup.
Rapordaki 6/2/3 yerine 5/2/3 (07 kullanılamadığı için).

`python scripts/14_make_pairs.py data/UAV_VisLoc_dataset --check 50` → `data/pairs/pairs_v1.csv`, 19.778 satır,
sha256 `docs/checksums/pairs_v1.sha256`; iki koşuda birebir aynı.

| Küme | pos | hard | easy |
|---|---|---|---|
| train | 3.498 | 3.354 | 3.498 |
| calib | 1.063 | 868 | 1.063 |
| test | 2.177 | 2.080 | 2.177 |

- Küçük haritalı bölgelerde zor negatif eksik: 05 (195 eksik), 06 (97), 10 (hepsi; harita ≈ 1,6 × 1,5 km,
  1,4 km karodan uzak nokta yok). Harita kenarındaki karelerde pozitif karonun bir kısmı harita dışında
  (05 ve 10'da medyan %70 harita içi).
- 50 rastgele çiftin görsel kontrolü: `results/faz1/pairs_check/` — GT ayak izi, yön ve ölçek tutarlı.
- Kalibrasyonun iki bölgesi de "tarım" → Mondrian (arazi türüne göre) kalibrasyonda tek grup verir.
