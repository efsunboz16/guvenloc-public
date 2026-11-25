# Faz 2 — Modül 1: kuzey kanonikleştirici (İP2)

> Dal `faz2/m1`. Plan: MIMARI.md "Faz 2". Hedef (H1, ablasyon satırı 4): medyan yön hatası ≤ 10°, kare başına
> RoMa v2 çağrısı ≤ 1,5, başarı@25 satır 2'den (calib 0,319, 4 çağrı) anlamlı biçimde düşük değil.

## Kararlar (26.11)

| # | Karar | Gerekçe |
|---|---|---|
| F2-1 | M1'in yön hatası **calib bölgelerinde (05, 11)** ölçülür; calib etiketleri eğitimde kullanılmaz | Plan calib'de yön etiketi olmadığını varsayıyordu; Faz 1'de var olduğu görüldü. Test bölgeleri Faz 6'ya kadar kapalı |
| F2-2 | Doğrulama uçuşu (λ, öğrenme oranı seçimi) önceden konan kuralla seçilir: grubunu başka eğitim bölgesiyle paylaşmayan, ≥ 500 kareli adaylardan etiket gürültüsü (medyan \|d\|) en küçük olan → **04** | 08/09 aynı coğrafi grup (sızıntı); 10 yalnızca 144 kare; 01'in etiketi gürültülü (9,5°) |
| F2-3 | Eğitim bölgeleri **01, 08, 09, 10** (2.760 kare); doğrulama 04 (738 kare) | Plan "6 eğitim bölgesi" diyordu; 07 dışlandığı için eğitim kümesi 5 bölge |
| F2-4 | Eğitim/doğrulama etiketi **φ_corr = Phi1 + c(bölge, uçuş yönü)**; c = o yöndeki güvenilir karelerde RoMa yönü − Phi1 farkının medyanı (n ≥ 20, değilse c = 0). Calib (değerlendirme) etiketleri ham Phi1 | Etiket gürültüsü uçuş yönüne bağlı ve sistematik (2A); düzeltme bacaklar arası çok tutarlı (aşağıda). Plandan sapma: etiket yalnız Phi1 değil, Phi1 + RoMa'dan bölge/yön düzeltmesi |

## 2A. Etiketler ve etiket gürültüsü (`scripts/50_m1_labels.py`)

Etiket φ = Phi1 mod 360 (görüntü üst kenarının pusula yönü, Faz 1B), 36 bölme (10°) →
`data/m1/labels_v1.csv` (sha256 `docs/checksums/m1_labels_v1.sha256`). Gürültü: RoMa v2 homografisinin verdiği
görüntü yönü ile φ farkı d, yalnızca eşleşmesi güvenilir karelerde (satır 2 kabul + konum hatası < 25 m).

| bölge | küme | kare | güvenilir | medyan d | medyan \|d\| | \|d\| > 10° |
|---|---|---|---|---|---|---|
| 01 | train | 817 | 330 | −9,5° | 9,5° | %48 |
| 04 | val | 738 | 161 | −0,2° | 2,2° | %0 |
| 08 | train | 1033 | 140 | +7,5° | 18,0° | %92 |
| 09 | train | 766 | 164 | +2,0° | 2,4° | %1 |
| 10 | train | 144 | 7 | +3,1° | 3,2° | (n küçük) |
| 05 | calib | 473 | 137 | −1,4° | 3,8° | %4 |
| 11 | calib | 590 | 202 | −1,1° | 1,2° | %0 |

**Bulgu: gürültü rastgele değil, uçuş yönüne bağlı ve sistematik.** Uçuşlar gidiş-dönüş bacaklarından oluşuyor;
fark aynı yöndeki bacaklarda neredeyse sabit, yön değişince değişiyor:
- 08: φ ≈ 110° bacaklarında d ≈ −22…−25°, φ ≈ 278° bacaklarında d ≈ +15…+17° (5'er bacak, her biri ±2° içinde).
- 01: φ ≈ 30° bacaklarında d ≈ −12…−15°, φ ≈ 167° bacaklarında d ≈ −3…−6°.
- 04, 09: iki yönde ±2…4° (küçük ama aynı işaret deseni).

Desen, Phi1'in kameranın gerçek yönünü değil uçuş rotasını (yere göre gidiş yönü) verdiği, aradaki farkın da
yan rüzgârla burun açısı (crab) olduğu açıklamasıyla uyumlu: rüzgâr aynıyken gidiş ve dönüşte farkın işareti
değişir. Bunun üstüne ≈ −3° sabit kayma var (Faz 1'de de görülmüştü). Calib bölgelerinde (05, 11) fark küçük
(medyan |d| 1–4°) → değerlendirme etiketleri temiz.

Eğitim karelerinin %36'sında etiket hatası 10°'nin, yani hedef doğruluğun üstünde. Etiketler olduğu gibi
kullanılırsa M1 rotayı öğrenir; 08 gibi bölgelerde ±20° tutarsız hedef görür.

## 2A (devam). Etiket düzeltmesi — karar F2-4 (`data/m1/labels_v2.csv`)

| bölge | yön (medyan φ) | düzeltme c |
|---|---|---|
| 01 | 167° / 30° | −4,5° / −13,5° |
| 04 (val) | 351° / 169° | +1,1° / −4,3° |
| 08 | 111° / 278° | −24,0° / +15,6° |
| 09 | 312° / 127° | +2,6° / −2,1° |
| 10 | 169° / 326° | 0 / 0 (güvenilir kare 7 < 20 → ham) |

**Dürüst sınama (birini-dışarıda-bırak):** her bacağın düzeltmesi yalnızca aynı yöndeki diğer bacaklardan
hesaplanıp o bacakta ölçüldü (33 bacak, 795 güvenilir kare): medyan |etiket hatası| **5,2° → 1,4°**, 10°'yi
aşan kare oranı **%36 → %1** (08: 18,0° → 1,7°; 01: 9,5° → 2,3°). Düzeltme bacaktan bacağa genelleniyor.

Kalan risk: 10'un etiketleri ham (144 kare, eğitimin %5'i; gürültüsü ölçülemedi). Düzeltme RoMa'nın ölçtüğü
yönü referans alıyor; RoMa yönü ile drone'un gerçek yönü arasındaki fark Faz 1B'de < 1° bulunmuştu.
Ayrıntı: `docs/sonuclar/faz2/label_correction.md`.

## 2B. Ağ — `guvenloc/m1_canon.py`

- **Kurulum:** escnn 1.0.11. `lie_learn`'ün PyPI paketi numpy 1.x ile derlenmiş (numpy'ı 1.26'ya indiriyordu);
  numpy 2.5.4'e karşı GitHub'dan (`edf012f`) derlendi, `py3nj` de derlendi; numpy/scipy değişmedi (komutlar
  `requirements.txt`'te). Sunucuda `python3.12-dev` gerekli.
- **Mimari:** C8 (`rot2dOnR2(N=8)`) eşdeğer CNN, regular alanlar 16/32/64/96/128, ilk katman 7×7 adım 2, sonraki
  bloklar 3×3 + antialiased ortalama havuzlama; **1,01 M parametre**. Son katman tek regular alan (8 yön logiti) →
  uzaysal ortalama → bant sınırlı Fourier interpolasyonu (frekans ≤ 4) ile 36 bölme → öğrenilen logit ölçeği.
  θ̂ = argmax + parabolik alt-bölme inceltmesi.
- **Girdi:** merkez kare (kısa kenar) → 321 px önbellek; eğitimde GPU'da keyfi açıyla döndürme + **257×257**
  dairesel maske. Kenarın tek sayı olması şart: 256'da adımlı katmanların örnekleme ızgarası simetrik olmadığından
  90°'de bile eşdeğerlik bozuluyor (ölçüldü); 257'de tam.
- **Dönme kuralı:** içerik saat yönünde α döndürülünce φ' = φ − α; kare φ̂ kadar saat yönünde döndürülünce kuzey-yukarı.
- **Eşdeğerlik (`tests/test_m1_equivariance.py`):** 90°'nin katlarında 36 bölmeli çıktı tam 9k bölme kayıyor
  (rastgele ağırlıklı ağda θ̂ sapması < 0,14°). 45°'nin tek katları ve ara açılar piksel ızgarasında tam olamaz
  (bilineer yeniden örnekleme); plan "45°'nin katlarında tam" diyordu — doğrusu: 90°'de tam, 45°'de yaklaşık.
  45° ve 20° sapması eğitilmiş ağda ölçülüp raporlanacak (2E).
- **Hız (3090):** 64'lük grup eğitim adımı 195 ms, 3,0 GB → epoch (2.760 kare) ≈ 9 s.
