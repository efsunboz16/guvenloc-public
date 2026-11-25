# Faz 1 — Durum raporu (24 Kasım 2026)

> Tek sayfada: ne bitti, hangi sorunlar çıktı ve nasıl çözüldü, hangi kararlar bekliyor (önerilerle),
> ne kaldı. Kalan işlerin komutları: [faz1_kalan_isler.md](faz1_kalan_isler.md).
> Dal: `faz1/altyapi` — PR #2. Çalışılan makine: RTX 5050 Laptop 8 GB (24.11'e kadar).
> **25.11'den itibaren bütün işler üniversite sunucusunda (RTX 3090 24 GB, Ubuntu 22.04);** kurulum doğrulandı
> (köşe hatası 0,32 px, testler yeşil).

## 1. Bitenler

| Adım | Ne yapıldı | Ayrıntı |
|---|---|---|
| 1A | Tam paket doğrulandı (11 bölge, 6.774 kare), checksum listesi | [faz1_veri_notlari.md](faz1_veri_notlari.md) |
| 1B | Coğrafi altyapı (`io_visloc`, `geo`); yön tanımı ve ölçek RoMa ile sayısal olarak doğrulandı | aynı belge |
| 1C | `configs/split.yaml` v1 (dondurulmuş), `pairs_v1.csv` (uyarlamalı karo, 19.973 çift, birebir yeniden üretilebilir) | aynı belge |
| 0.4 | Hassasiyet birimi çözüldü: P son refiner aşamasının ızgara pikselinde | [faz1_matcher_onbellek.md](faz1_matcher_onbellek.md) |
| 1E | `matcher.py`, `cache.py`, önbellek üretimi (kesintiye dayanıklı), doğrulama/onarım | aynı belge |
| 1F | Metrikler, istatistik testleri, taban çizgileri 1–3 (calib tablosu, `base` önbelleğiyle) | [faz1_tekrarlama.md](faz1_tekrarlama.md) |
| 1D | WxBS v1.1 mAA@10px **55,91** (rapor 55,4 ✅); hız/bellek ölçümü; MegaDepth duman testi (Kaggle T4) orijinalle birebir | aynı belge |
| K2 | Karo deneyi: uyarlamalı karo kazandı ve uygulandı | §3 K2 |

Testler: 73 test (GPU'lular dahil), CI'da ruff + pytest yeşil.

**Calib taban çizgisi tablosu (özet, eski `dev_base` önbelleği, sabit 1400 m karo — 3090'da `cache/v1` ile yenilenecek):**

| Satır | Yöntem | başarı@25 | AUROC | FAR-neg |
|---|---|---|---|---|
| 1 | 0° + RANSAC | 0,198 | 0,950 | 0,042 |
| 2 | 4 yön + en çok inlier | **0,291** | 0,935 | 0,026 |
| 3 | 4 yön + en düşük entropi | 0,151 | **0,968** | 0,048 |

## 2. Karşılaşılan sorunlar ve çözümleri

### Veri seti
| Sorun | Çözüm |
|---|---|
| 09'un uydu haritası 4 parça | `io_visloc.satellite_source`: VRT mozaiği, bölge tek harita gibi okunur |
| 09'un koordinat CSV'si 52 m'ye kadar kayık | Georeferans her zaman tif'in kendisinden okunur |
| 6.774 kare (makalede 6.742) | Veri tutarlı (CSV ↔ dosyalar birebir); fark belgelendi |
| Rapor "Phi Yunnan ve Shandan'da yok" diyordu; gerçekte yalnızca 07'de yok | Belgelendi; MIMARI'ye not düşüldü |
| 07 kullanılamaz (30 kare, yön yok, harita ≈ 50 m şerit) | Ayrımdan çıkarıldı → 5/2/3 |
| 08 ve 09 aynı bölge (Huzhou-3), haritalar üst üste | `groups` ile aynı kümede; birbirine kolay negatif olamaz |
| 06'da 6 karenin GPS'i bozuk (harita dışı, zeminin altında) | `excluded_frames` |
| Yön sütununun anlamı belirsiz | RoMa ile ölçüldü: Phi1 = görüntü üst kenarının pusula yönü (saat yönü), fark ≈ −3° |
| EXIF yok; `height` bazı bölgelerde deniz seviyesine göre | Ayak izi = 0,980 × AGL (RoMa ile); bölge zemin yükseklikleri `configs/pairs.yaml` |

### Yazılım / RoMa v2
| Sorun | Çözüm |
|---|---|
| Hassasiyet birimi bilinmiyordu (σ ~1000 kat yanlış) | Kaynak kod + deney: ızgara pikseli; `matcher.py` görüntü pikseline çevirir |
| romav2 `sample()` tek başına 7,9 GB → mega1500 8 GB kartta 50 s/çift | `balanced_sample`: aynı algoritma, parça parça KDE (aynı tohumla birebir aynı); mega1500 4,4 s, 7,3 GB |
| romav2 `sample()` tek yönlü ayarlarda yanlış hassasiyet döndürüyor | Hassasiyet yoğun haritalardan okunuyor |
| Uydu karosu kesmek 3 s | Yeni kesme: 7–12 kat hızlı, kayma < 0,1 px |
| Kovaryans fp16'da taşıyor / negatif çıkıyor | Σ float32 saklanıyor |
| romav2 WxBS benchmark'ı görselleştirmede çöküyor | Görselleştirme devre dışı (ölçümü etkilemez) |
| WxBS mAA tanımı belirsizdi (61,5 mi 55,9 mu) | Makaledeki tanım 0–10 px (11 eşik) |
| Bilgisayar çöktü, önbellek yarıda kaldı | Kaldığı yerden devam + `31_verify_cache.py` (2 bozuk dosya temizlendi) |
| `runs.csv` git dışındaydı, geçersiz satırlar vardı | Git'e alındı, temizlendi |

## 3. Kararlar

K2 ve K6 karara bağlandı (24.11). K1 kuralı 3090'da uygulanacak. K3 PR #2'de onay bekliyor. K4 ve K5'te itiraz gelmezse önerilen uygulanır.

### K1. GT konumu görüntü merkezi değil (15–40 m sistematik kayma)
CSV `lat/lon` drone'un GPS konumu; görüntü merkezi kamera eğimi kadar kayık (yanal ↔ Omega r = −0,88,
ileri ↔ Kappa r = +0,58). Başarı@10'u düşürüyor.
- **Öneri:** Ana değerlendirmede CSV kalır (literatürle karşılaştırılabilirlik; conformal kalibrasyon bu
  hatayı güven dairesine katar). Tam önbellek gelince eğim modeli YALNIZCA eğitim bölgelerinde binlerce
  kareyle yeniden kurulur. **Önceden konan kural:** calib'de medyan kaymayı ≥ %30 azaltırsa düzeltilmiş GT
  ikincil sonuç olarak (`pairs_v2`) raporlanır; azaltmazsa konu kapanır.
- Önbelleği ETKİLEMEZ.

### K2. Karo boyutu — KARAR: uyarlamalı (uygulandı)
Sabit 1400 m'yi yalnızca bölge 11 gerektiriyor; alçak bölgelerde drone ayak izi karonun 1/3'ü kalıyor ve
RoMa'nın ölçek farkı büyüyor (30 karelik denemede 1000 m biraz daha iyiydi, anlamlı değil).
- **Öneri:** Kaydırma ±200 m kalır. Karo kareye göre uyarlanır: `karo = ayak izi köşegeni + 2 × kaydırma`
  (≈ 750–1330 m; ölçek oranı bölgeler arasında benzer, drone karesi her zaman sığar).
  Önbellekten önce deney: sabit 1400 m vs uyarlamalı, eğitimden 300 kare, önbellekle aynı RoMa ayarı,
  başarı@25 (mega1500 ≈ 3 sa, base ≈ 0,5 sa). Eğitimde iyi olan seçilir; test/calib'e dokunulmaz. Sonuçla danışmana gidilir.
- Kod: `configs/pairs.yaml: tile_mode` (`fixed` | `adaptive`), `scripts/15_tile_experiment.py`.
- **Sonuç (24.11, eğitim, 300 kare, `base`, retsiz):** uyarlamalı karo (medyan 1040 m) satır 2 başarı@25'i
  0,170 → **0,213** (+0,043 [0,000, 0,089], McNemar 22'ye 9, p = 0,029); satır 1'de 0,100 → 0,160 (p = 0,0005).
  5 eğitim bölgesinin 4'ünde iyileşme (10'da 0,12 → 0,08). Kural sonucu: **uyarlamalı**.
  Tablo: [sonuclar/faz1/k2_tile.md](sonuclar/faz1/k2_tile.md).
- **Uygulandı (24.11):** `tile_mode: adaptive`, `pairs_v1.csv` yeniden üretildi (19.973 çift; checksum güncel).
  Gerekçe: MIMARI §1C'deki karo kuralının (ayak izi + 2 × kaydırma) kare başına uygulanması; yalnızca eğitimde,
  önceden konan kuralla seçildi. Danışman özetinde belirtilecek.

### K6. Önbellek ayarı — KARAR: `base`
Tam mega1500 önbelleği dizüstünde ≈ 105 sa, Kaggle T4'te ≈ 70 sa. `base` ile bütün kümelerde aynı ayar
(eşikler eğitimden calib/test'e taşınır). mega1500 yalnızca MegaDepth tekrarlamasında. Raporda donanım kısıtı
olarak belirtilir.
- **Not (25.11):** Gerekçe 8 GB dizüstü / Kaggle kotasıydı. 3090'da mega1500 tek çift 0,54 s (aynı testte dizüstünde 2,1 s);
  tam mega1500 önbelleği sunucuda kabaca bir güne sığabilir. Karar değiştirilmedi; `speed` adımının ölçümüyle
  yeniden açılıp açılmayacağına ekip karar verir (değişirse K2 deneyi `base` ile yapıldığı için not düşülmeli).

### K3. INTERFACES §1–2 değişiklikleri
Σ float32 (P yerine; fp16 gerçek dosyalarda taştı/bozuldu), çift listesine ek sütunlar (geriye uyumlu).
- **Öneri:** Olduğu gibi onay. PR #2'de "INTERFACES onayı" yorumuna dört kişi 👍 (MIMARI §3.1 kuralı).

### K4. Kalibrasyonun iki bölgesi de tarım → arazi Mondrian'ı imkânsız
- **Öneri:** Ayrım değişmez (dondurulmuş, raporla tutarlı). Faz 4'te Mondrian gruplaması arazi yerine
  σ̂ dilimlerine göre (`group_by="sigma_quantile"`) — "en zor çeyrek" ölçütüyle de uyumlu. Arazi kırılımı
  raporda; calib (tarım) → test (kent/dağlık) kayması H3'ün açıkça sınadığı risk olarak raporlanır ve
  MIMARI §13'e eklenir.

### K5. Taban çizgisi başarısı düşük (en iyi başarı@25 = 0,29)
Hata değil: eşleşme ya tutuyor (5–50 m) ya tamamen kaçıyor (tarım/su, mevsim farkı) + K1'deki GT kayması.
- **Öneri:** Tablo `cache/v1` ile yenilenir; bölge başına drone/uydu tarih farkı belgelenir;
  Faz 1 raporuna "zor durumlar" paragrafı. Düşük taban M2–M4'ün katkısını göstermek için iyi başlangıç.

## 4. Kalanlar — hepsi 3090'da (25.11)

Ayrıntılı kılavuz ve komutlar: [faz1_kalan_isler.md](faz1_kalan_isler.md). Özet:

1. `python scripts/faz1_3090.py`: kurulum kontrolü, veri ve çift checksum'ı, MegaDepth duman testi,
   önbellek calib + train, taban çizgisi tablosu, MegaDepth-1500 + C4 (5'er tohum), önbellek seq + test (≈ 20 sa)
2. Claude ile: K1 eğim analizi ve `sigma_scale` kalibrasyon betikleri (yazılacak), belgelerin doldurulması
3. PR #2 → `main`, `v0.0`, danışmana özet

Kaggle denendi (T4'te duman testi orijinalle birebir, 2,9 sa/tohum) ama kullanılmadı:
[faz1_kaggle.md](faz1_kaggle.md).

## 5. Faz 2'ye devreden notlar

- Bazı uçuş bacaklarında Phi1, RoMa ile ölçülen yönden 25°'ye kadar sapıyor (01, 08) → M1 etiket gürültüsü
  (MIMARI 2A'daki kontrol bunu ölçmeli).
- `matcher.match` keyfi açıyla döndürmeyi destekliyor (M1'in θ̂'sı için hazır, testli).
- Calib bölgelerinde de yön etiketi var; MIMARI kararı gereği M1 eğitiminde kullanılmıyor.
