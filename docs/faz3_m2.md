# Faz 3 — Modül 2: olasılık hacmi (İP3)

> Dal `faz3/m2`. Plan: MIMARI.md "Faz 3". Hedef (H2, ablasyon satırı 5): aynı başarı@25 düzeyinde yanlış kabul
> oranı RANSAC + inlier eşiğinden anlamlı biçimde düşük; AUROC yüksek. İşlemsel tanım ve ön-kayıt: F3-3,
> `configs/h2.yaml`.

## Kararlar (26.11)

| # | Karar | Gerekçe |
|---|---|---|
| F3-1 | θ araması **M1'den bağımsız**: tam çember 5° (σ₀ = 12 m) → kaba tepe ±6° / 1° (σ₀ = 4 m). Plandaki "M1 ±15°" kurulumu kodda (`theta_center`) ve ablasyonda | M1-C karelerin ≈ %25'inde yanlış çeyrekte (Faz 2D); RoMa v2 dönmeye dayanıklı olduğundan eşleşmeler yönü kendileri taşıyor. M1'e bağlamak M2'yi M1'in hatasıyla sınırlardı |
| F3-2 | Eşleşme kovaryansı **izotrop**: σ_i² = σ₀² + iz(Σ_map)/2; σ dilimlerine göre saçma + Gauss bulanıklaştırma; σ_i > 30 m atılır | İyi eşleşmelerde Σ_B ≈ 1–3 m (σ₀'ın altında), anizotropi etkisi küçük; tam 2×2 pencereli splat 10–50 kat yavaş. Plandan sapma |
| F3-3 | H2 ön-kaydı (`configs/h2.yaml`): satır 4a (M1-C tek çağrı, RANSAC + inlier) vs 5a (aynı eşleşmeler, hacim tepesi + S). Yanlış kabul = kabul ∧ ¬(pozitif ∧ hata ≤ 25 m). Eşik eşleme calib'de (aynı başarı@25). (i) yanlış kabul McNemar p < 0,05, (ii) AUROC farkı GA alt sınırı > 0. İkisi birden | Tek çağrı kurulumunda iki yöntem aynı eşleşmeleri görüyor: fark yalnız konum + skor. Sıralı politikalarda (4c/4e) durma kuralı skora bağlı, karşılaştırma karışır → ikincil |
| F3-5 | **H2 eşleme kuralı kusurlu (kuru denemede bulundu):** her yöntem kendi konumunu kullandığından başarı tavanları farklı; 5a'nın tavanı (tümü kabul 0,233) 4a'nın hedefinin (0,247) altında → "≥ hedef olan en büyük eşik" yok, kural her şeyi kabul ediyor. `h2.yaml` henüz değiştirilmedi; nihai M2 (v0 ya da inceltici) seçildikten sonra, **test öncesi** yeni kararla düzeltilecek | Kural sonuca bakılarak değil, tanımsız kaldığı için düzeltilecek; ölçütler (i, ii) aynı kalacak |
| F3-6 | **Öğrenilmiş inceltici (3D) tasarımı:** girdi v0 tepesi çevresinde 2 × 13 θ × 48 × 48 (±48 m) kırpıntı (c ağırlıklı oy + ağırlıksız yoğunluk) + küresel v0 metrikleri; küçük 3B U-Net (364 bin parametre) → hücre logitleri + "boş" logiti, softmax tümü üzerinde. Kayıp: pozitif ∧ GT kırpıntıda → θ-marjinal konum NLL; negatif ∨ GT dışarıda → −log p(boş). Eğitim 01, 08, 09, 10; erken durdurma ve ana skor seçimi (S25 = argmax çevresinde 25 m kütlesi, ya da Snull = 1 − p(boş)) val 04'te. Karar kuralı plandaki gibi: calib'de v0'ı AUROC ve başarı@25'te geçmezse ana yöntem v0 | Tam hacim (13 × 500² × 41 bin) diske sığmıyor; doğru/yanlış kararı tepe çevresinde veriliyor. "Boş" sınıfı planın "negatiflerde kütleyi düşüren terimi"nin olasılıkçı karşılığı. Yön kaybı yok: Phi1 yön etiketi gürültülü (GT yönü çoğu zaman ±6° dışında) |
| F3-7 | **3D sonucu: inceltici karar kuralını geçmiyor, ana M2 yöntemi v0 kalır.** Calib'de AUROC 0,951 (v0 0,940), ama başarı tavanı 0,217 (v0 0,233); kural ikisini birden istiyor. Ek tohum çalıştırılmadı | Tek tohumla bile tavan farkı (−1,6 puan) ve aynı-etiket AUROC farkı (−0,002 [−0,012; +0,009]) sonucu değiştirecek büyüklükte değil; ek tohum yalnız aynı sonucu doğrulardı. Kural sonuçtan önce yazılmıştı (F3-6) |
| F3-8 | **F3-5 düzeltmesi, H2 ön-kaydı h2_v2:** "doğru" iki yöntemde de ortak konum kestiriminden (satırın dönmesinde RANSAC merkezi) hesaplanır; eşleme kuralı aynı (4a'nın başarı@25'i). Böylece iki yöntemin tavanı aynı olur ve kural her zaman tanımlı. 2v için hedef tavanı aşarsa hedef = tavan (işaretli). Hacim tepesinin konum başarısı ayrı bilgi satırında. S modelinin eğitim hedefi değiştirilmedi (hacim tepesi etiketi); ölçütler (i, ii) aynı | H2 ret kararını sınıyor; 3B–3C teşhisinde konumun iki yöntemde eşdeğer olduğu (≤ 25 m 0,186 / 0,181), farkın 25 m sınırındaki K1 gürültüsü olduğu yalnız eğitim verisiyle gösterilmişti. Kendi-etiket karşılaştırması kalitesiz konumu ödüllendiriyor (3D'de görüldü). Kural tanımsız kaldığı için değiştirildi; yön (v0 lehine mi, aleyhine mi) bilinmeden seçildi, calib'de iki etiketle de v0 4a'nın altında |
| F3-4 | Hacim metrikleri test bölmesinde Faz 6'ya kadar çıkarılmaz (`60_m2_features.py` reddeder) | Ön-kayıt disiplini |

## 3A. Oylama hacmi v0 — `guvenloc/m2_volume.py`, `configs/m2.yaml`

`m_i(θ) = v_i − s·R(θ)(u_i − u₀)`, `R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]]`, θ = `gt_heading` kuralı.
İşaret gerçek çiftte RANSAC homografisiyle doğrulandı (04_0300_pos: inlier'larda m_i ile H merkezi arası medyan
3,4 px; ters işarette 76 px). Izgara 2 m, karo kapsamı (≈ 430–660 hücre kenar). Çift × dönme başına ≈ 60 ms
(3090, önbellekten okuma dahil).

Sentetik testler (`tests/test_m2_volume.py`, 7): tepe 2 m / 1° içinde (0°, 37°, 181,5°, 300°), M1 kipinde de;
negatifte destek < pozitifin %10'u, ρ ve entropi büyük; saçma kütleyi koruyor.

Gerçek çiftlerde ilk bakış:

| çift | tepe θ (GT / RANSAC) | tepe − RANSAC merkezi | destek | ρ | H (bit) |
|---|---|---|---|---|---|
| 04_0300_pos (9.123 inlier) | 167,5° (171,4° / 167,7°) | 2 m | 9.388 | 0,005 | 12,4 |
| 11_0300_pos | 90,2° (90,6° / 90,2°) | 10 m | 7.186 | 0,10 | 13,5 |
| 01_0200_pos (RoMa kaçırıyor) | — | 391 m | 51 | 0,39 | 15,9 |
| 05_0200_hard0 (negatif) | — | — | 10 | 0,05 | 17,2 |
| 04_0300_easy0 (negatif) | — | — | 2 | 0,81 | 18,8 |

**Not (K1):** 04_0300_pos'ta hacim ve RANSAC aynı yeri gösteriyor ama GT'den 50 m uzakta. Faz 1 K1 kayması
(GT = drone GPS'i, görüntü merkezi değil): çok güvenilir eşleşmelerde (≥ 4.000 inlier) medyan konum hatası
bölgeye göre 18–31 m (04'te 30,7 m). "Doğru" etiketi bu yüzden gürültülü; iki yöntemi aynı ölçüde etkiler.

## 3B–3C. Hacim metrikleri ve H2 kuru denemesi — `scripts/60_m2_features.py`, `scripts/61_m2_score.py`

Çıkarım: calib 12.392 + train 41.400 (çift, dönme), 724 + 1.897 s (≈ 46 ms/dosya). S: lojistik (IRLS,
C = 1), eğitim bölmesinin tüm (çift, dönme) satırları; hedef "doğru" (pozitif ∧ hacim tepesi ≤ 25 m).

**Calib (`docs/sonuclar/faz3/h2_calib.md`):**

| satır | başarı@25 | yanlış kabul | FAR-neg | AUROC [GA] |
|---|---|---|---|---|
| 4a M1-C 1 çağrı · RANSAC + inlier (τ = 709) | 0,247 | 0,144 | 0,063 | 0,944 [0,932; 0,955] |
| 5a aynı eşleşmeler · hacim + S | 0,233* | 0,920* | 1,000* | 0,936 [0,923; 0,949] |
| 5a, yalnız destek | 0,233* | — | — | 0,940 |
| 5a, yalnız ρ / yalnız H | 0,233* | — | — | 0,912 / 0,920 |
| 2 4 yön · RANSAC + inlier | 0,319 | 0,148 | 0,055 | 0,927 |
| 2v 4 yön · hacim + S | 0,293* | 0,900* | 1,000* | 0,928 |

\* Eşik eşleme kurulamadı (F3-5): hacim tepesinin başarı tavanı hedefin altında, kural tümünü kabul ediyor.
Bu satırların yanlış kabul sayıları anlamsız; H2 (i) ve (ii) ✗ (AUROC farkı −0,008 [−0,021; +0,005]).

**Teşhis (yalnız eğitim bölgeleri, calib'e bakmadan):**
- **Konum:** hacim tepesi RANSAC kadar iyi. Tüm pozitif (çift, dönme) ≤ 25 m: hacim 0,186 / RANSAC 0,181;
  ≥ 2.000 inlier'da 0,570 / 0,580, tepe − RANSAC merkezi medyan 2,1 m (p90 8,6 m), drone ekseninde
  sistematik kayma yok (ileri 0,3 m, sağ 0,1 m). Calib'deki 1,4 puanlık tavan farkı 25 m sınırında kıl payı
  yer değiştiren kareler (K1 kayması 18–31 m olduğu için sınır çok kalabalık).
- **Ret skoru:** bölge-dışarıda çapraz doğrulama (5 eğitim bölgesi), aynı etiket (RANSAC konumuyla doğru):
  AUROC S **0,962** = inlier **0,962**, yalnız destek 0,954. Hacim etiketiyle S 0,959 / inlier 0,937, ama
  bu fark etiketlerin yönteme göre değişmesinden geliyor, skordan değil.

**Sonuç:** eğitimsiz hacim v0 ne konumda ne ret kararında RANSAC + inlier sayısının ötesine geçiyor.
Bu beklenebilirdi: v0'ın en güçlü metriği "destek" (25 m içindeki c-ağırlıklı oy) inlier sayısının yumuşak
bir sürümü; ρ ve H ek bilgi katmıyor (katsayılar destek tarafından baskın). H2'yi sağlayacak olan, planda
zaten sıradaki adım: **öğrenilmiş inceltici (3D)**. Hacmin uzaysal yapısını (tepe biçimi, ikincil tepeler,
eşleşme yoğunluğu) doğrusal olmayan biçimde kullanabilir. v0 ana tabloda karşılaştırma satırı olarak kalır.

## 3D. Öğrenilmiş inceltici — `scripts/62_m2_crops.py`, `scripts/63_m2_refiner.py`

Kırpıntılar: `data/m2/crops_{train,calib}.npy` (2 × 13 × 48 × 48, float16). Eğitim tohum 0, 15 epoch
(≈ 43 s/epoch, 3090); val 04'te en iyi epoch **2**, ana skor **S25** (val AUROC 0,948; Snull 0,945).
Sonraki epoch'larda eğitim kaybı düşmeye ve val konum tavanı yükselmeye devam ediyor (0,159 → 0,173), ama
val AUROC düşüyor (0,946 → 0,931): ağ konumu iyileştirdikçe ret skorunu kaybediyor.

**Calib (`docs/sonuclar/faz3/refiner_calib.md`):**

| satır | AUROC (kendi etiketi) | AUROC (RANSAC etiketi) | başarı tavanı |
|---|---|---|---|
| 4a RANSAC + inlier | 0,944 | 0,944 | 0,253 |
| 5a_v0 hacim + S | 0,940 | 0,940 | 0,233 |
| 5r inceltici (S25) | 0,951 | 0,942 | **0,217** |

- AUROC farkı 5r − 4a: kendi etiketiyle +0,008 [−0,003; +0,018], **aynı etiketle −0,002 [−0,012; +0,009]**.
  Kendi etiketindeki kazanç, inceltici daha kötü konumlandırdığı için etiketlerin değişmesinden geliyor
  (daha az, daha kolay pozitif); skorun kendisinden gelmiyor.
- Eşit kabul oranında (%22,9) yanlış kabul 4a 0,144 / 5r 0,155, başarı@25 0,247 / 0,216; McNemar p = 0,078
  (yalnız 4a'da yanlış kabul 148, yalnız 5r'de 181).

**Karar (F3-7):** kural (v0'ı AUROC **ve** başarı@25'te geçmek) sağlanmıyor. Ana M2 yöntemi v0 kalır.
İnceltici tabloda deneme satırı olarak durur.

**Neden olmadı (yorum):**
1. Konum etiketi K1 kayması yüzünden gürültülü (medyan 18–31 m). ±48 m'lik kırpıntıda 25 m'lik hedef
   ağın düzeltebileceğinden çok sınır gürültüsü taşıyor.
2. Ret bilgisinin çoğu zaten inlier sayısında ve destekte (3B–3C teşhisi). Kırpıntının uzaysal yapısı
   val'de ölçülebilir ek bilgi katmıyor; ağ 2 epoch'ta doyuyor.
3. H2'nin bu tasarımla sağlanması beklenmiyor. Test öncesi açık kalan iş F3-5: eşleme kuralının düzeltilmesi.
   Bu düzeltme v0 için yeni bir karar olarak yazılacak.

## F3-5 düzeltmesi — H2 kuru denemesi, h2_v2 (`docs/sonuclar/faz3/h2_calib.md`)

Ortak etiketle (RANSAC merkezi ≤ 25 m) eşleme artık kuruluyor:

| satır | başarı@25 | yanlış kabul | FAR-neg | AUROC [GA] |
|---|---|---|---|---|
| 4a RANSAC + inlier (τ = 709) | 0,247 | **0,144** | 0,063 | **0,944** [0,932; 0,955] |
| 5a hacim + S | 0,247 | 0,343 | 0,272 | 0,924 [0,909; 0,938] |
| 5a, yalnız destek | 0,247 | 0,176 | 0,071 | 0,940 |
| 2 4 yön · RANSAC + inlier | 0,319 | 0,148 | 0,055 | 0,927 |
| 2v 4 yön · hacim + S (hedef tavana indirildi) | 0,316 | 0,758 | 0,819 | 0,915 |

McNemar 5a vs 4a: yalnız 4a'da yanlış kabul 83, yalnız 5a'da 699 (p ≈ 1e−122). AUROC farkı
−0,020 [−0,033; −0,006]. **H2 (i) ✗, (ii) ✗**: v0 hacmi ret kararında RANSAC + inlier'ı geçmiyor,
calib'de anlamlı biçimde gerisinde.

S'nin hacim-tepesi etiketiyle eğitilmiş olması (ön-kayıtta öyle; değiştirilmedi) ortak etikette bir
uyumsuzluk yaratıyor. Ancak yalnız eğitim bölgeleriyle yapılan bölge-dışarıda teşhiste (3B–3C), ortak
etiketle eğitilmiş S de inlier sayısıyla yalnız berabere kalıyordu (0,962 = 0,962). Yani S hedefi
değiştirilse de H2'nin sağlanması beklenmiyor. Bunu calib'e bakarak yeniden ayarlamıyorum.

**Faz 3 durumu:** eşleme kuralı artık test için tanımlı (h2_v2). v0 da inceltici de (F3-7) calib'de H2'yi
desteklemiyor. Test (Faz 6) ön-kayıtla aynen koşulacak; beklenen sonuç H2'nin reddi. M2'nin olası değeri
ret skorunda değil, yayılma/çok tepe gibi belirsizlik çıktılarında (MIMARI'deki "karar için olasılık
çıktısı" kullanımı).
