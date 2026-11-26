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

## 2C. Eğitim — ilk deneme (v1): ezberliyor, genellemiyor (`scripts/51_m1_prep.py`, `scripts/52_m1_train.py`)

Kurulum: eğitim 01/08/09/10 (2.760 kare, φ_corr), her örnek U(0, 360°) döndürülür; dairesel Gauss yumuşatmalı
çapraz entropi (σ = 1 bölme), AdamW, 60 epoch, grup 64, bf16. λ·L_prior kullanılmadı: denetimli kurulumda önselin
hedefi (kuzey-yukarı) etiketin kendisi, ayrı bir şey ayarlamıyor (plandan sapma). Öğrenme oranı {3e-4, 1e-3, 3e-3}
doğrulamada (04) seçildi → 3e-3; 3 tohum. İzleme: `scripts/m1_izle.py`. Koşu 45 dk.

| küme | bölge | kare | medyan hata (0°) | hata < 10° | hata deseni |
|---|---|---|---|---|---|
| eğitim | 01, 08, 09, 10 | 2.760 | ≈ 0° | %100 | ezber |
| doğrulama | 04 | 738 | 59–64° | %7 | çoğu ±90° civarında, sistematik |
| calib | 05 | 473 | 19° | %23 | medyan −18° |
| calib | 11 | 590 | 96–111° | %2 | ±90° ve 180° civarında |
| **calib toplam (3 tohum)** | | 1.063 | **50,9 ± 1,5°** | %11 | hedef ≤ 10° ✗ |

Eşdeğerlik (eğitilmiş ağ, calib): 90° tam (0,0°), 45° medyan 3,0°, 20° medyan 4,4° sapma.

**Teşhis:** eğitim kaybı yumuşatılmış hedefin alt sınırına (≈ 1,42) iniyor ve eğitim karelerinin %100'ü 10° içinde
→ ağ her eğitim sahnesini tanıyıp onun kuzeyini ezberliyor. 4 uçuş (her biri tek yer, iki uçuş yönü) yeni bölgeye
taşınan bir "kuzey ipucu" (gölge yönü, yol/bina düzeni) öğretmeye yetmiyor; yeni bölgedeki hata rastgele değil,
sistematik (±90° kümeleri) — ağ başka bölgenin ipucunu yanlış yöne bağlıyor. Hiperparametre (lr) farkı küçük
(63–67°); tohumlar arası fark küçük → sorun veri/görev kaynaklı, ayardan değil.

## 2C-v2. Öneri A (sert düzenleme) ve B (uydu kırpıntıları): ikisi de işe yaramadı

`python scripts/52_m1_train.py ab --lr 3e-3 --epochs 40` (tohum 0, her 5 epoch'ta val 04 → erken durdurma);
uydu kırpıntıları `scripts/53_m1_sat_prep.py` (yalnız eğitim bölgeleri 01/08/09/10, eğitim karesi başına 3,
GT çevresinde ≤ 300 m, kenar = drone ayak izi × U(0,8; 1,25), φ = 0; 8.280 kırpıntı). İzleme:
`scripts/m1_izle.py --ab`. Seçim yalnız val'e göre; calib her yapılandırma için **teşhis** amaçlı. Koşu ≈ 1 sa.

| yapılandırma | ne | en iyi epoch | val 04 (12 açı) | calib (0°) | 05 | 11 | calib < 10° |
|---|---|---|---|---|---|---|---|
| v1_es | v1 + erken durdurma | 15 | **56,8°** | 55,4° | 27,5° | 95,2° | %7,7 |
| A1_sert | renk/doygunluk/ölçek/bulanıklık oynatma | 5 | 68,8° | 70,8° | 16,7° | 114,8° | %14,3 |
| A2_sert_gri | A1 + gri ton | 30 | 65,4° | 62,2° | 15,6° | 124,0° | %15,6 |
| A3_sert_gri_kucuk | A2 + küçük ağ, wd 1e-3 | 25 | 65,9° | 60,1° | 14,5° | 124,2° | %17,2 |
| B1_sert_uydu50 | A1 + yığının yarısı uydu | 30 | 71,1° | 49,4° | 22,8° | 80,1° | %15,2 |
| B2_sert_gri_uydu50 | B1 + gri ton | 20 | 70,1° | 42,6° | 68,5° | 40,8° | %15,7 |
| B3_yalniz_uydu | yalnız uydu | 30 | 64,2° | 84,1° | 101,3° | 52,5° | %9,3 |

Şans düzeyi (düzgün rastgele tahmin): medyan hata 90°, < 10° oranı %5,6; 90°'nin katlarına göre (mod 90)
< 10° oranı %22. Belirsizlik çözümlemesi (`results/faz2/m1/m1_ab_ambiguity.json`, docs/sonuclar kopyası):

- **Doğrulama 04'te bütün yapılandırmalar şans düzeyinde:** < 10° oranı %2–9 (şans %5,6), mod 90 %21–29
  (şans %22). Yani "4 yönden biri doğru" türünde bir belirsizlik bile öğrenilmemiş — M1, satır 2'nin 4 dönmesini
  daraltacak bilgi de vermiyor.
- **Calib'deki küçük farklar bölgeye özgü yanlılık:** 05 ve 11 yapılandırmadan yapılandırmaya yer değiştiriyor
  (B2: 05 68°, 11 41°; A3: 05 15°, 11 124°). Calib toplamındaki 42–84° aralığı gürültü; hiçbiri hedefin
  (≤ 10°) yakınında değil.
- **Uydu kırpıntıları öğreniliyor ama taşınmıyor:** B1/B2/B3 eğitim uydu kırpıntılarında %86–100 doğru
  (örneklem içi), drone karesinde şans düzeyi → uydu–drone alan farkı ve aynı 4 bölge.
- Erken durdurma tek başına val'i 64° → 57°'ye indirdi ama v1'de de 10. epoch'ta val 68° idi: ağ ezberlemeden
  önce de genellemiyordu.

**Sonuç (H1 için olumsuz bulgu):** Bu veriyle (4 eğitim uçuşu) tek dik bakışlı drone karesinden bölgeden bağımsız
mutlak kuzey öğrenilemiyor; ne düzenleme (A) ne eğitim bölgelerinin uydu haritası (B) bunu değiştiriyor. Sorun
"veri az ama görev öğrenilebilir" değil, "bu girdiden görev bu ölçekte öğrenilemez" tarafında. Sıradaki seçenekler:
C (M1'i drone karesi + uydu karosu → göreli dönme modeline çevirmek; H1 tasarımını değiştirir, danışmana
danışılmalı) veya D (olumsuz sonucu raporlayıp satır 2'nin 4 dönmesiyle devam).

## 2C-C. Öneri C fizibilitesi: göreli dönme modeli genelliyor (kolay kurulumda)

Model `M.C8RelRot` (1,04 M parametre): drone karesi ve kuzey-yukarı uydu karosu aynı C8 eşdeğer gövdeden
geçer (ağırlık paylaşımı); her biri 32 regular alan → uzaysal ortalama → (32, 8) betimleyici; 8 grup kayması
üzerinden kosinüs korelasyonu → 36 bölmeye Fourier interpolasyonu. Drone dalında eşdeğerlik tam; işaret
`tests/test_m1_equivariance.py::test_relrot_sign` ile sınanıyor (yanlış işarette düşüyor).

Veri `scripts/54_m1c_prep.py`: karo kenarı = 1,5 × drone merkez karesinin ayak izi, merkez GT'den ≤ 0,25 ×
ayak izi kaydırılmış; eğitim karesi başına 3 karo (yalnız 01/08/09/10 haritası), val/eval karesi başına 1 karo
(04, 05, 11 haritası yalnız değerlendirmede — çıkarımda karo zaten verilir). Eğitim `scripts/55_m1c_train.py`
(tohum 0, 20 epoch × 130 adım, sert oynatma, her epoch val 04 → erken durdurma). İzleme `scripts/m1_izle.py --c`.
Koşu ≈ 1,5 sa.

| yapılandırma | en iyi epoch | val 04 (12 açı) | val < 10° | calib (0°) | 05 | 11 | calib < 10° | calib < 45° |
|---|---|---|---|---|---|---|---|---|
| C1 (lr 1e-3) | 18 | 4,5° | %78 | 7,8° | 17,2° | 4,4° | %55 | %73 |
| **C2 (lr 3e-3) — val'e göre seçilen** | 17 | **4,4°** | %80 | **7,7°** | 12,9° | 5,1° | %56 | %73 |
| C3 (lr 1e-3, gri) | 20 | 4,7° | %79 | 6,2° | 15,7° | 3,4° | %63 | %80 |

- **Mutlak kuzey modeline göre fark:** calib 43–84° → 6–8°; val 57–71° → 4,4–4,7°. Calib medyanı hedefin
  (≤ 10°) altında, ama yalnız kolay kurulumda ve tek tohumda.
- **Kuyruk:** calib karelerinin %20–27'sinde hata > 45° (eşleşme bulunamayan kareler). Bunlar için RoMa'ya
  2 aday ya da satır 2'ye geri dönüş gerekecek — 2D'de güven/entropi eşiğiyle ele alınmalı.
- 05 (dağlık, 1.800 m+ irtifa) 11'den belirgin zor (13–17° vs 3–5°). Calib etiketleri ham Phi1 (düzeltilmemiş,
  bölge başına birkaç derece sapma olabilir) → gerçek hata biraz farklı olabilir.

**Sınırlar (sonraki adım):** sistemdeki karo ≈ 3× ayak izi ve merkez kayması ≤ 200 m. Karo büyüdükçe drone
karesi karonun küçük bir parçası olur ve küresel ortalama betimleyici zayıflar. Sıradaki deney: gerçek karo
boyutu (`tile_size_for`, adaptive) ile aynı modeli sınamak; gerekirse uydu dalında uzaysal maksimum
(drone betimleyicisini karo konumları üzerinde kaydırma) eklemek. Tasarım değişikliği danışman onayına bağlı.
