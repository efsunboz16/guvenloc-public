# Faz 2 — Modül 1: kuzey kanonikleştirici (İP2)

> **Durum: kapandı (26.11.2026, `v0.1`)** — H1 calib'de sağlanmadı, olumsuz ve açıklanmış; bkz. "Faz 2 özeti".
> Dal `faz2/m1`. Plan: MIMARI.md "Faz 2". Hedef (H1, ablasyon satırı 4): medyan yön hatası ≤ 10°, kare başına
> RoMa v2 çağrısı ≤ 1,5, başarı@25 satır 2'den (calib 0,319, 4 çağrı) anlamlı biçimde düşük değil.
> İşlemsel tanım ve ön-kayıt: karar F2-6, `configs/h1.yaml`. Modül 1 = göreli dönme (F2-5).

## Kararlar (26.11)

| # | Karar | Gerekçe |
|---|---|---|
| F2-1 | M1'in yön hatası **calib bölgelerinde (05, 11)** ölçülür; calib etiketleri eğitimde kullanılmaz | Plan calib'de yön etiketi olmadığını varsayıyordu; Faz 1'de var olduğu görüldü. Test bölgeleri Faz 6'ya kadar kapalı |
| F2-2 | Doğrulama uçuşu (λ, öğrenme oranı seçimi) önceden konan kuralla seçilir: grubunu başka eğitim bölgesiyle paylaşmayan, ≥ 500 kareli adaylardan etiket gürültüsü (medyan \|d\|) en küçük olan → **04** | 08/09 aynı coğrafi grup (sızıntı); 10 yalnızca 144 kare; 01'in etiketi gürültülü (9,5°) |
| F2-3 | Eğitim bölgeleri **01, 08, 09, 10** (2.760 kare); doğrulama 04 (738 kare) | Plan "6 eğitim bölgesi" diyordu; 07 dışlandığı için eğitim kümesi 5 bölge |
| F2-4 | Eğitim/doğrulama etiketi **φ_corr = Phi1 + c(bölge, uçuş yönü)**; c = o yöndeki güvenilir karelerde RoMa yönü − Phi1 farkının medyanı (n ≥ 20, değilse c = 0). Calib (değerlendirme) etiketleri ham Phi1 | Etiket gürültüsü uçuş yönüne bağlı ve sistematik (2A); düzeltme bacaklar arası çok tutarlı (aşağıda). Plandan sapma: etiket yalnız Phi1 değil, Phi1 + RoMa'dan bölge/yön düzeltmesi |

| F2-5 | **Modül 1 yeniden tanımlandı (öneri C):** "tek drone karesinden mutlak kuzey" yerine **drone karesi + kuzey-yukarı uydu karosu → göreli dönme** (`M.C8RelRot`). Çıktı (φ, 36 bölme), etiketler, C8 eşdeğer gövde aynı; karo çıkarımda zaten verilir | Mutlak kuzey bu veriyle öğrenilemedi (2C, 2C-v2: şans düzeyi); göreli dönme genelliyor (2C-C). Plandan sapma — danışmana bilgi notuyla iletildi |
| F2-6 | **H1 ölçütleri değişmedi**, yalnız işlemselleştirildi ve test öncesi `configs/h1.yaml`'a ön-kaydedildi: (i) M1-C medyan yön hatası ≤ 10° (pozitif çiftler, 0°); (ii) doğru karoda kare başına ortalama RoMa çağrısı ≤ 1,5; (iii) başarı@25 farkı (satır 4 − satır 2) için 50 karelik blok bootstrap %95 GA alt sınırı > −0,03 (eşdeğersizlik payı 3 puan). Üçü birden | Modül değişti diye başarı ölçütünü değiştirmek hedefi sonuca göre kaydırmak olurdu. "Anlamlı biçimde düşük değil" McNemar p ≥ 0,05 olarak okunursa düşük güç ödüllendirilir → açık bir pay (δ = 0,03) ile eşdeğersizlik testi |
| F2-7 | Ana tabloda **satır 4 = 4d** (M1-C ardışık, en çok 2 dönme; H1 bu satırda sınanır); **4c** (kabul edilince dur, en çok 4) ikincil çalışma noktası olarak aynı tabloda | 4d önerideki çağrı bütçesine uyan politika; 4c "aynı başarı, daha az çağrı" sonucunu gösterir. İkisi de calib görülmeden tanımlandı, eşikleri eğitimden |
| F2-8 | "Tek dik bakışlı kareden mutlak kuzey bu veriyle (4 eğitim uçuşu) öğrenilemiyor" bulgusu raporda H1 altında **olumsuz sonuç** olarak yer alır (2C, 2C-v2 tabloları, şans düzeyi karşılaştırması) | Olumsuz sonuç bilgi taşıyor ve C'ye geçişin gerekçesi |
| F2-9 | v0.1 M1-C modeli = **D2_olcek_rot90** (karesi başına 4 karo, uydu ölçek U(0,85; 1,18) + 90° dönme oynatması), tohum **s2** (val 04 en iyi, 8,8°); calib sonuçları 3 tohum ortalaması ± ss ile raporlanır | Seçim ön-kayıtlı kurala göre (val 04, 12 açı medyanı). Calib'de C5'in 4d başarısı biraz daha yüksek (0,282 vs 0,270 ± 0,007) ama calib'e göre seçmek ön-kaydı bozar |
| F2-10 | H1 ön-kaydı v2 (`configs/h1.yaml`): H1 satırı **4d → 4e** (güvene bağlı bütçe). 4e'nin tek parametresi q, calib'den önce yalnız val 04'te seçilir: pozitif kare başına çağrı ≤ 1,45 altında başarı@25 en büyük. Ölçütler (i–iii) aynı. 4d ve 4c aynı tabloda kalır | 4d sabit 2 çağrılık bütçeyle M1-C'nin emin olmadığı karelerde doğru dönmeye ulaşamıyordu. Satır değişikliği calib kuru denemesinden sonra yapıldı (calib geliştirme kümesi); test bölgelerine dokunulmadı. Commit `a231e75`, calib koşusundan önce |

Kararlar F2-5…F2-10 kullanıcı onayıyla alındı (26.11); danışmana not ile bildirildi.

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

### 2C-C (devam). Gerçek karo boyutu: calib 11,8°, sınırda

`python scripts/54_m1c_prep.py --gercek` (val/eval: pairs_v1 pozitif karosunun kendisi — ayak izi köşegeni +
400 m ≈ 3× merkez karesi, merkez kayması ≤ 200 m, medyan 142 m; eğitim: o karo + aynı kuralla 1 rastgele;
513 px, 7.321 karo) → `python scripts/55_m1c_train.py --veri gercek --epochs 20 --batch 32` (64 bellek
yetmedi). Uydu karosu döndürülmez/maskelenmez; renk oynatması var, ölçek oynatması yok. İzleme
`scripts/m1_izle.py --cg`. Koşu ≈ 2,5 sa.

| yapılandırma | en iyi epoch | val 04 (12 açı) | val < 45° | calib (0°) | 05 | 11 | calib < 10° | calib < 45° |
|---|---|---|---|---|---|---|---|---|
| C4 küresel ortalama | 12 | 10,7° | %73 | 11,6° | 13,8° | 8,7° | %46 | %72 |
| **C5 kayan pencere (5 hücre) — val'e göre seçilen** | 5 | **10,2°** | %77 | **11,8°** | 18,7° | 5,7° | %46 | %74 |

- Kolay kuruluma göre (calib 7,7°) belirgin düşüş ama hâlâ mutlak kuzeyin (43–84°) çok önünde; calib medyanı
  hedefin (≤ 10°) biraz üstünde. Kayan pencere küçük bir kazanç (val < 45° %73 → %77), medyanda fark yok.
- Doğrulama eğrisi gürültülü (C4: epoch 9–20 arası 10,7–14,7°; C5 en iyisine 5. epoch'ta ulaşıyor) →
  tohum varyansı ölçülmeli; tek tohum.
- **Güven (`results/faz2/m1/m1c_gercek_guven.json`):** tahmin ±45° penceresindeki olasılık kütlesi güven
  ölçüsü olarak kullanıldı. C5, calib'de en güvenli %50: < 45° %85,5, medyan 9,5°; en güvenli %80: < 45° %79,5.
  Güven kuyruğu ayırıyor ama zayıf: tek çağrıya güvenmek her güven düzeyinde %15+ hatalı dönme demek.
- **2D için sonuç:** M1'in en olası dönmesiyle RoMa'yı çağırıp, RoMa reddederse kalan dönmelere düşmek
  (ardışık strateji) beklenen çağrıyı ≈ 1 + 0,26 × 3 ≈ 1,8'e indirir (RoMa'nın doğru dönmede de reddettiği
  kareler hariç) — satır 2'nin 4 çağrısına karşı. Hedef ≤ 1,5 için M1'in < 45° oranı ≈ %83+ olmalı.

Sonraki iyileştirme adayları: eğitimde daha çok karo (karesi başına 2 → 5+), tohum ortalaması, uydu tarafına
ölçek oynatması, pencere boyutunu bölge irtifasına göre ayarlamak (karo/ayak izi oranı 2,6–3,7).

## 2D. M1-C → RoMa bağlama (satır 4), önbellekten simülasyon — `scripts/56_m1c_roma.py`

Yeni RoMa çağrısı yok: Faz 1F'de her calib çifti × 4 dönme için çıkarılan RoMa özellikleri
(`results/faz1/baselines/feats_v1_calib_px5.csv`) kullanıldı. M1-C (C5 kayan pencere, gerçek karo) her calib çifti
için (1.063 pozitif + 2.035 zor/kolay negatif; negatif karolar `data/m1/calib_pairs_513.npy`) 4 dönmenin
olasılığını verir (φ ∈ [r − 45°, r + 45°) kütlesi; önbellekte r = drone'a saat yönünde dönme). Satır 4a/4b/4d
eşikleri eğitim negatiflerinden (FAR-neg ≤ 0,05; 4a: 4 dönme havuzu → τ = 709, 4b/4d: komşu dönme çiftlerinin
maksimumu → τ = 1.055); 4c satır 2 eşiğini (1.462) kullanır. Calib'de ayar yapılmadı. GA %95, 1.000 tekrar.

| satır | yöntem | RoMa çağrısı/kare (poz/neg) | başarı@25 [GA] | başarı@50 | FAR-neg |
|---|---|---|---|---|---|
| 1 | RoMa 0° | 1 | 0,214 [0,176; 0,252] | 0,412 | 0,068 |
| 2 | 4 yön + en çok inlier | 4 | **0,319** [0,280; 0,361] | 0,575 | 0,055 |
| 3 | 4 yön + en düşük entropi | 4 | 0,163 | 0,313 | 0,063 |
| 4a | M1-C tek çağrı | 1 | 0,253 [0,208; 0,301] | 0,466 | 0,067 |
| 4b | M1-C ilk 2 dönme | 2 | 0,283 | 0,516 | 0,066 |
| **4c** | M1-C ardışık, τ₂ geçince dur | 3,35 (2,32/3,89) | **0,316** [0,275; 0,361] | 0,572 | 0,055 |
| **4d** | M1-C ardışık, en çok 2 | **1,79 (1,48/1,96)** | 0,282 [0,239; 0,331] | 0,513 | 0,066 |

McNemar (başarı@25, pozitifler, Holm): 4a > 1 (79 vs 37, p = 2,4e-4); 4d > 1 (103 vs 30, p = 5,8e-10);
4d < 2 (9 vs 48, p = 4,6e-7); **4c ≈ 2 (5 vs 8, p = 0,58)**; 4a < 2 (p = 5e-14).

- **Aynı başarı, daha az çağrı (4c):** satır 2 ile istatistiksel olarak aynı başarı; pozitif karede 4 → 2,32
  çağrı (−%42). Reddedilen kareler (negatiflerin çoğu, pozitiflerin ~%36'sı) yine 4 çağrı yapıyor → genel
  ortalama 3,35. Çalışma sırasında karelerin çoğu pozitif (doğru karo) olduğundan gerçek kazanç 2,3'e yakın.
- **Bütçe ≤ 2 (4d):** pozitifte 1,48 çağrı (öneri hedefi ≤ 1,5), başarı 0,282: satır 1'den +6,8 puan,
  satır 2'den −3,7 puan.
- **Tek çağrı (4a):** satır 1'den anlamlı iyi (+3,9 puan), aynı maliyet.
- Bölgeler tutarlı (05 ve 11'de aynı sıra). FAR-neg: eğitimde seçilen eşikler calib'de 0,055–0,068 (satır 1
  ve 2'deki kaymayla aynı düzey).
- **Henüz ölçülmedi:** M1-C'nin kendi süresi (513 px karo + 257 px kare, 1 M parametre) — RoMa çağrısına
  göre küçük olmalı, 2E'de ölçülecek. Tek tohum.

Sonuç: M1-C, RoMa çağrı sayısı–başarı eğrisini satır 1–2 arasında içeriden geçiriyor: aynı başarıda ~%40
daha az çağrı (4c) ya da hedef bütçede (≤ 1,5 çağrı) satır 1'e göre anlamlı kazanç (4d). Satır 2'nin
başarısını ≤ 1,5 çağrıda yakalamak için M1-C'nin < 45° oranının (%74) artması gerek.

### 2D (devam). H1 ön-kaydının calib'de kuru denemesi (C5 modeli)

`configs/h1.yaml` ile `scripts/56_m1c_roma.py`: başarı@25 farkı (satır − 2), blok bootstrap %95 GA —
4c: −0,003 [−0,008; +0,005] (satır 2 ile eşdeğer, pay içinde); 4d: −0,037 [−0,051; −0,023].
Satır 4d'de: (i) yön hatası 11,9° ✗; (ii) 1,48 çağrı ✓; (iii) GA alt sınırı −0,051 ✗ → **H1 calib'de sağlanmıyor**.
4d'nin açığı M1-C'nin < 45° oranından (%74) geliyor; iyileştirme turu (karesi başına 4 karo, uydu ölçek/90°
oynatması, 3 tohum) bunu hedefliyor. Test kararı Faz 6'da, bu ölçütlerle.

### 2D (devam). İyileştirme turu (v2): yön hatası düştü, RoMa başarısı değişmedi

`python scripts/54_m1c_prep.py --gercek --ek 3` (eğitim karesi başına 4 karo, 12.841 karo, CPU'da tutuluyor) →
`scripts/55_m1c_train.py --veri gercek4` (`CONFIGS_V2`, 12 epoch, batch 32), zincir `logs/m1c_v2_zincir.sh`:
aşama 1 D1/D2 tohum 0 → val'e göre seçilen yapılandırma tohum 1, 2. Sonra her tohum için `56_m1c_roma.py`.

| yapılandırma | tohum | val 04 (12 açı) | calib (0°) | 05 | 11 | calib < 10° | calib < 45° |
|---|---|---|---|---|---|---|---|
| C5 (önceki, 2 karo) | 0 | 10,2° | 11,8° | 18,7° | 5,7° | %46 | %74 |
| D1 ölçek | 0 | 10,4° | 13,4° | 18,7° | 9,5° | %43 | %64 |
| **D2 ölçek + 90° (seçilen)** | 0 | 9,4° | 9,8° | 15,0° | 6,3° | %50 | %75 |
| D2 | 1 | 8,9° | 10,3° | 13,3° | 7,0° | %49 | %75 |
| D2 | 2 | **8,8°** | 10,2° | 16,1° | 6,2° | %49 | %76 |
| **D2, 3 tohum** | | 9,0 ± 0,3° | 10,1 ± 0,3° | | | %50 | %75 |

Satır 4, calib (RoMa önbelleği; `results/faz2/m1/roma_row4_D2_olcek_rot90*_calib.md`), D2 3 tohum ortalaması ± ss:

| | C5 | D2 (3 tohum) |
|---|---|---|
| M1-C medyan yön hatası (pozitif çiftler) | 11,9° | **10,1 ± 0,2°** (9,9 / 10,3 / 10,2) |
| M1-C < 45° (pozitif) | %74,0 | %75,2 ± 0,6 |
| 4a başarı@25 (1 çağrı) | 0,253 | 0,248 ± 0,008 |
| 4c başarı@25 (2,35 çağrı/poz) | 0,316 | 0,311 ± 0,001 |
| 4d başarı@25 (1,49 çağrı/poz) | 0,282 | 0,270 ± 0,007 |
| 4d − satır 2, GA alt sınırı | −0,051 | −0,079 / −0,060 / −0,061 |
| 4c − satır 2 [GA] | −0,003 [−0,008; +0,005] | −0,008 [−0,017; +0,001] (her tohumda benzer) |

H1 (satır 4d, `configs/h1.yaml`): (i) 9,9° ✓ / 10,3° ✗ / 10,2° ✗ — eşikte; (ii) 1,49 ✓ (üç tohumda);
(iii) ✗ (üç tohumda). **H1 calib'de sağlanmıyor.**

- **Yön hatası ≠ RoMa başarısı.** Medyan 1,8° düştü ama RoMa başarısı yerinde: RoMa dört 90° dönmeden biriyle
  çağrılıyor ve kendi içinde ±45°'ye kadar dönmeyi tolere ediyor; başarıyı belirleyen M1-C'nin **doğru çeyreği
  seçme oranı** (< 45°), o da %74 → %75'te kaldı. Medyanı iyileştiren oynatmalar zaten doğru çeyrekteki
  kareleri keskinleştirdi, yanlış çeyrekteki %25'lik kuyruğu küçültmedi.
- 4d'nin satır 2'ye açığı (≈ 4,5–5 puan) tamamen bu kuyruk: 4d 2 çağrıda duruyor, M1-C'nin ilk iki tahmini
  dışında kalan doğru dönmeyi bulamıyor. 4c (kabul edilene kadar devam) satır 2 ile eşdeğer kalıyor.
- Tohum varyansı küçük (4c ± 0,001, 4d ± 0,007): sonuç tek tohumun şansı değil.
- Calib'de C5 4d'de biraz daha iyi, ama model seçimi ön-kayıt gereği val'e göre (F2-9).

Sonuç: M1-C'yi daha da keskinleştirmek H1 (iii)'ü kurtarmaz; açığı kapatacak şey yanlış çeyrek kuyruğu.
İki yol: (a) M1-C'nin çeyrek doğruluğunu hedefleyen kayıp/veri (ör. 90°-sınıflandırma başlığı, daha çok bölge),
(b) güvene bağlı bütçe (M1-C emin değilse 3.–4. dönmeye izin; ortalama çağrıyı ≤ 1,5'te tutacak eşik
eğitim/val'den seçilir). (b) yeni bir politika olduğundan H1 satırını değiştirmez; ancak ön-kayıt güncellemesi
(yeni karar) ile ve testten önce satır 4 yapılabilir.

### 2D (devam). Güvene bağlı bütçe (4e) ve kahin tavanı (4k): H1 (iii) bu RoMa ile ulaşılamaz

**4e (karar F2-10).** M1-C'nin dönme olasılıkları sıralanır. Sıralı toplamı ≥ q olan en az k dönme (1–4)
bütçe olur. Bu dönmeler sırayla denenir, ilk inlier ≥ τ_k olanda durulur. τ_k eğitim negatiflerinden
seçilir: k dönmenin maksimumu için FAR-neg ≤ 0,05 (τ_1…τ_4 = 709 / 1.055 / 1.289 / 1.462). q, calib'den
önce val 04'te seçildi (`--sec`, `results/faz2/m1/q_sel_*.json`): tohum 0 / 1 / 2 için q = 0,90 / 0,925 /
0,95, val'de çağrı/poz 1,32 / 1,35 / 1,41. Val 04'te eğri çok düz: 1 çağrıda 0,194, 4 çağrıda 0,220.

**4k (kahin, yöntem değil).** 4d ile aynı politika, ama dönme sırası M1-C'den değil gerçek yönden gelir
(en yakın çeyrek önce). Kusursuz bir dönme kestiricisinin ≤ 2 çağrıda ulaşabileceği tavanı verir. Modelden
bağımsızdır.

| satır | çağrı/poz | başarı@25 (3 tohum) | − satır 2 [GA] |
|---|---|---|---|
| 1 RoMa 0° | 1 | 0,214 | |
| 2 4 yön | 4 | 0,319 | |
| 4a M1-C tek çağrı | 1 | 0,248 ± 0,008 | |
| 4c M1-C ardışık | 2,35 | 0,311 ± 0,001 | −0,008 [−0,017; +0,001] |
| 4d M1-C en çok 2 | 1,49 | 0,270 ± 0,007 | ≈ −0,05 [alt −0,06…−0,08] |
| **4e M1-C güvene bağlı** | **1,38** (1,35 / 1,38 / 1,40) | 0,267 ± 0,006 | ≈ −0,05 [alt −0,06…−0,08] |
| **4k kahin, en çok 2** | 1,45 | **0,297** | **−0,022 [−0,032; −0,011]** |

H1 (v2, satır 4e), calib: (i) 9,9° ✓ / 10,3° ✗ / 10,2° ✗; (ii) 1,35–1,40 ✓; (iii) ✗ → **sağlanmıyor**.

- 4e, 4d ile aynı başarıyı %7 daha az çağrıyla veriyor. Açığı kapatmıyor: M1-C'nin güveni, yanlış çeyrekteki
  kareleri yeterince ayırmıyor (2C-C'de de zayıftı).
- **Asıl bulgu: kusursuz M1-C bile (iii)'ü geçemiyor.** 4k'nin GA alt sınırı −0,032 < −0,03; McNemar 4k <
  satır 2, p = 8,8e-4. Neden: RoMa v2 bu veride dönmeye büyük ölçüde dayanıklı. Pozitif (çift, dönme)
  başına başarı@25 (τ = 709):

  | dönme, gerçek yöne göre | başarı@25 | medyan inlier |
  |---|---|---|
  | doğru çeyrek (< 45°) | 0,275 | 1.833 |
  | komşu (45–135°) | 0,221 | 716 |
  | ters (> 135°) | 0,214 | 676 |

  Doğru dönme inlier sayısını 2,5 kat artırıyor ama başarıyı yalnız ≈ 5–6 puan. Herhangi bir dönmede
  başarılı olan 377 karenin 85'inde doğru çeyrek başarısız, başka bir dönme başarılı. Satır 2'nin
  satır 1'e üstünlüğü (+10,5 puan) yaklaşık yarı yarıya iki kaynaktan geliyor: doğru yönü bulmak ve
  **dört bağımsız deneme**. İkincisi RoMa'nın dönmeye bağlı rastgeleliğinden geliyor; hiçbir dönme
  kestiricisi bunu ≤ 1,5 çağrıda karşılayamaz.
- M1-C, kusursuz bir kestiricinin satır 1'e göre kazancının (0,297 − 0,214 = 8,3 puan) ≈ %67'sini yakalıyor
  (4d: 5,6 puan).

**Sonuç ve H1'in durumu.** Ölçütler ön-kayıtlı; değiştirilmiyor. H1 testte büyük olasılıkla sağlanmayacak.
Bunun nedeni M1-C'nin kalitesi değil, (iii)'ün varsayımı: "RoMa'ya doğru dönme verilirse dört yönlü
taramanın başarısı korunur". Bu varsayım RoMa v2 için tutmuyor. Raporda H1 sonucu böyle verilecek: 4k
tavanıyla birlikte, olumsuz ama açıklanmış. Testte de raporlanacak, ön-kayıt gerektirmeyen ikincil
bulgular:
(a) aynı maliyette (1 çağrı) satır 1'den anlamlı kazanç (4a, +3,4 puan);
(b) satır 2 ile eşdeğer başarıda çağrıyı 4'ten 2,35'e indirme (4c);
(c) ≤ 1,5 çağrıda kusursuz kestiricinin tavanına yakınlık (4e/4d ile 4k karşılaştırması).

## 2E. Değerlendirme: ara açı eşdeğerliği ve hız — `scripts/57_m1c_2e.py`

Çıktı `results/faz2/m1/m1c_2e.md`. v0.1 modeli `D2_olcek_rot90_s2` (F2-9); eşdeğerlik 3 tohumda ölçüldü.

**Ara açı eşdeğerliği (2B'de verilen söz).** Calib pozitiflerinde drone karesi saat yönünde α döndürüldü.
Tahmin tam olarak −α kaymalı; ölçülen sapma |φ̂(α) − (φ̂(0) − α)|, 3 tohum:

| α | medyan sapma | > 5° | > 45° (çeyrek atlaması) | v0.1 yön hatası medyanı |
|---|---|---|---|---|
| 90°, 180°, 270° | 0,00° | %0 | %0,0–0,6 | 10,1–10,2° |
| 45°, 135°, 315° | 1,9–2,4° | %22–28 | %8–10 | 10,2° |
| 10° | 3,2–3,6° | %31–35 | %9 | 10,9° |
| 20°, 30°, 200° | 4,6–5,0° | %46–50 | %10–12 | 12,2–12,6° |

- 90°'nin katlarında tam (yapı gereği, 2B ile aynı). 45°'nin katlarında yaklaşık: C8 grubunun elemanı ama
  piksel ızgarasında tam değil. Grup dışı açılarda (20°, 30°) medyan sapma ≈ 5°, yön hatası medyanı
  10,2° → 12,6°.
- **Kuyruk önemli:** karelerin ≈ %9–12'sinde küçük bir dönme bile (10°) tahmini > 45° atlatıyor. Bunlar
  dağılımı çok tepeli, yani kararsız kareler. 2D'deki yanlış çeyrek kuyruğunun (%25) bir kısmı bu
  kararsızlık; tek bir dönmüş kopyadan değil, birkaç açının ortalamasından karar vermek (test zamanı
  dönme ortalaması) bu kareleri yumuşatabilir. Denenmedi; Faz 2 sonrası aday.
- Gerçek çalışmada drone yönü keyfi olduğundan ilgili sayı ara açılardaki hatadır. Calib'in kendi yönleri
  zaten keyfi; H1 (i)'deki 10,1° bunu içeriyor (0° = kare olduğu gibi).

**Hız (RTX 3090, tek kare, ısınma sonrası 100 tekrar medyanı).**

| hassasiyet | M1-C toplam | karo gövdesi (513 px) | kare gövdesi (257 px) | p90 | ek bellek |
|---|---|---|---|---|---|
| bf16 | **5,8 ms** | 3,4 ms | 1,9 ms | 6,6 ms | 0,10 GB |
| fp32 | 12,5 ms | 9,6 ms | 2,4 ms | 12,7 ms | 0,27 GB |

Ağırlıklar 4 MB. RoMa v2 `base` çağrısı 257 ms (Faz 1D), yani M1-C bir RoMa çağrısının ≈ %2'si.

**Kare başına beklenen süre** (calib çağrı sayıları × 257 ms + M1-C):

| satır | başarı@25 | pozitif kare | negatif kare | satır 2'ye göre hız (pozitif) |
|---|---|---|---|---|
| 1 RoMa 0° | 0,214 | 257 ms | 257 ms | ×4,0 |
| 2 4 yön | 0,319 | 1.028 ms | 1.028 ms | ×1 |
| 4a M1-C tek çağrı | 0,247 | 263 ms | 263 ms | ×3,9 |
| 4c M1-C ardışık | 0,310 | 610 ms | 1.004 ms | **×1,7** |
| 4d M1-C en çok 2 | 0,274 | 388 ms | 509 ms | ×2,6 |
| 4e M1-C güvene bağlı | 0,270 | 365 ms | 607 ms | ×2,8 |

Doğru karoda (çalışmadaki çoğunluk durumu), 4c satır 2 ile aynı başarıyı 1,7 kat hızlı veriyor. 4a, satır 1
ile aynı sürede +3,3 puan veriyor.

## Faz 2 özeti

- **Mutlak kuzey (plan):** öğrenilemedi, şans düzeyi (2C, 2C-v2). Olumsuz sonuç olarak raporlanır (F2-8).
- **M1-C, göreli dönme (F2-5):** calib medyan yön hatası 10,1 ± 0,2°, < 45° %75; 5,8 ms, 4 MB.
- **H1 (ön-kayıt v2, satır 4e):** calib'de (i) eşikte, (ii) ✓, (iii) ✗. Kahin tavanı (4k) gösteriyor ki
  (iii) RoMa v2'nin dönmeye dayanıklılığı nedeniyle hiçbir dönme kestiricisiyle ≤ 1,5 çağrıda
  sağlanamıyor. Test kararı Faz 6'da, bu ölçütlerle.
- **Kullanılabilir katkı:** aynı maliyette satır 1'e göre +3,3 puan (4a); satır 2 ile eşdeğer başarı
  1,7 kat hızlı (4c).
- MIMARI.md "Faz 2 bitti sayılır" koşulu (Hedef 2 ölçütleri) calib'de sağlanmıyor. Faz 2, H1 olumsuz ve
  açıklanmış olarak kapatılabilir; v0.1 = `D2_olcek_rot90_s2` + satır 4c/4e politikaları.
