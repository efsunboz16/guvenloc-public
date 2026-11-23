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
