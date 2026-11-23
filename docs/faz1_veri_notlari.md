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
