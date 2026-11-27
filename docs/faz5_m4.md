# Faz 5 — Modül 4: zamansal füzyon (parçacık filtresi), ablasyon satırı 7

Hedef 5 / H4: medyan konum hatası tek kareye göre ≥ %20 azalır, kapsama ≥ %88 korunur. Test bölgeleri (02, 03, 06) Faz 6'da.

## Kararlar (26.11)

| # | Karar | Gerekçe |
|---|---|---|
| F5-1 | **H4 ön-kaydı** `configs/h4.yaml` (h4_v1), filtre sonuçlarından önce. Satır 6 (tek kare, kabulden bağımsız her pozitif kare) vs satır 7 (filtre ağırlıklı ortalaması); bütün pozitif kareler, filtre ilk kabul edilen karede başlar (öncesi satır 7 = satır 6). Ölçüt: (med6 − med7)/med6 ≥ 0,20 ve Wilcoxon tek yönlü p < 0,05; satır 7 kapsaması ≥ 0,88 (yarıçap = κ · ağırlıklı %90 yayılım, κ calib'de). Hiper-parametreler yalnız eğitim bölgelerinde, κ ve γ calib'de | MIMARI'deki "medyan hata tek kareye göre" tanımsızdı: hangi kareler, kabul edilmeyenler ne olur. Tek kare satırı her karede bir konum veriyor (`Decision.xy`), filtre de; aynı kare kümesinde eşleştirilmiş karşılaştırma en yalın tanım |
| F5-2 | **GT gürültü tabanı ölçüldü; ölçüt değişmez.** Ardışık kare çiftlerinde (iki kare de uyduyla iyi eşleşmiş, n = 2.339) yer değiştirme farkları medyanı: görüntü↔uydu 13 m, görüntü↔GPS 26 m, uydu↔GPS 32 m. GPS (GT) kare başına ≈ 20 m bağımsız gürültü taşıyor (K1 bulgusuyla, 18–31 m, tutarlı) | Tek kare kabul edilen pozitiflerde medyan hata 28–31 m (calib) zaten bu tabana yakın; filtrenin kazancı ancak yanlış tepeli ve reddedilen karelerden gelebilir (tüm karelerde tek kare medyanı 05: 121 m, 11: 54 m). Raporlanacak sınırlılık |
| F5-3 | **Filtre tasarımında üç düzeltme (yalnız eğitim verisine bakılarak, calib'den önce):** (a) odometri kapısı: adım uzunluğu son 5 iyi adımın medyanının 0,5–2 katı dışındaysa son iyi adım kullanılır (sabit hız, B planı); (b) θ ölçümü sağlam karışım (1 − 0,3) N(θ − θ̂; σ) + 0,3; (c) sensör sıfırlaması: kabul edilmiş ölçüm bulutu açıklamıyorsa (ağırlıklı ortalama P/maxP < eşik) filtre o tek kare kararından yeniden başlar | İlk sürüm eğitimde tek kareden kötüydü (01: 30 → 155 m, 09: 427 → 3.463 m). Nedenler kare kare izlendi: kabul edilmiş yanlış tepelerin θ̂'sı 80–170° sapıyor ve yönü bozuyordu; dokusuz/tekrarlı zeminde homografi "takılıyor" (01'de adımların %3,8'i < 0,3 × GPS adımı); dönüşte kopan bulut karonun dışına düşünce ε tabanı onu geri getiremiyordu (hata 57 km'ye kadar) |
| F5-4 | **Filtre hiper-parametreleri eğitimde seçildi** (`scripts/72_m4_filter.py --tune`, 162 ayar, ölçüt havuzlanmış medyan hata): λ = 2, σ_pos = 5 m + 0,1 × adım, ε = 10⁻³, θ ölçümü σ = 5°, sıfırlama eşiği 10⁻⁴, N = 2.000, sıçrama p = 0,02 / 100 m. Eğitimde satır 6 → 7: 337 → 39 m | Bölge başına 01 30 → 27, 04 47 → 37, 08 471 → 172, 09 427 → 42, 10 401 → 322 m. 08 komşu ayarlar arasında 152–324 m oynuyor (kararsız) |
| F5-5 | **κ = 3,25 sabitlendi** (calib 05+11 havuzu, tohum 0; satır 7 yarıçapı = κ · ağırlıklı %90 yayılım). Faz 6 testi `tune.json` ayarı, tohum 0 ve bu κ ile, yeniden ayar yok | Tohum 0–4: azalma 0,373–0,384, κ 3,00–3,25 → sonuç tohuma duyarsız. κ'nın bölgeler arası taşınması H3'teki FAR gibi açık risk: 05'te seçilen κ 11'de 0,947, 11'de seçilen 05'te 0,789 kapsama |
| F5-6 | **5C γ seçim kuralı (sonuçtan önce):** tahmin adımından sonra parçacık yayılımı (ağırlıklı RMS uzaklık, m) ≤ γ ise o karede uydu eşleştirmesi (M1 → RoMa → M2 → M3) yapılmaz, yalnız tahmin. Izgara γ ∈ {0 (her kare), 25, 50, 75, 100, 150, 200, 300, 500} m, calib 05+11, filtre ayarı ve κ = 3,25 sabit. Seçilen γ: medyan hata ≤ 1,05 × med7(γ = 0) ve kapsama ≥ 0,88 koşulunu sağlayan EN BÜYÜK γ (en çok atlama); hiçbiri sağlamazsa γ = 0. H4 kararı γ = 0 (satır 7) ile; 5C ikincil satır 7-γ | Hesap kazancı ile hata arasındaki ödünleşimi MIMARI 5C'deki gibi calib'de seçmek; kural %5 hata payına izin veriyor, kapsama ölçütünü korumak şartıyla |
| F5-7 | **γ = 50 m seçildi** (F5-6 kuralı, calib): uydu eşleştirmesi karelerin %29,6'sında atlanır, medyan hata 38,3 → 39,8 m (+%4), kapsama 0,912. Faz 6'da ikincil satır 7-γ bu değerle | γ = 75'te hata +%12 (kural ✗); γ ≥ 150'de filtre tek kareden kötü |

## 5A. Uçuş dizileri (`scripts/70_m4_sequences.py`, `docs/sonuclar/faz5/seq_summary.md`)

Eğitim + calib bölgeleri, ardışık pozitif kareler (4.554 çift). Kareler sık: medyan 60–130 m / 3–7 s, örtüşme 0,69–0,82,
örtüşmesiz çift ≤ %0,3, atlama yalnız 1 kez (08). Drone↔drone homografisi (önbellek `cache/v1/seq`, 5 px) %99,9 çiftte
kullanılır (inlier ≥ 50, medyan 3.360–9.136).

| | değer |
|---|---|
| Δθ hatası (homografi vs ΔΦ1) medyan | 0,50° |
| hareket hatası (homografi vs GPS) medyan / %90 | 25,5 / 59,1 m (mesafeye oranı 0,27) |
| k kare toplanınca (k = 1 / 10 / 40) | 25 / 117 / 387 m |

Hareket hatasının çoğu GPS'ten (F5-2): uyduyla karşılaştırınca adım başına 13 m. Toplamda k ile büyüme tek bir ölçek ya da
dönme düzeltmesiyle kalkmıyor (bölge başına en küçük kareler ölçek/dönme denendi, uzun pencerede kötüleşti) → filtrede
hareket gürültüsü mesafeyle orantılı + taban, ölçüm güncellemesi kaymayı düzeltir. 10'da uydu konumları ardışık karelerde
sıçrıyor (uydu↔GPS 378 m), 08'de 57 m: bu bölgelerde tek kare tepeleri güvenilmez.

## 5B–5D. Kuru deneme: eğitimde ayarlanmış filtre → calib (`docs/sonuclar/faz5/h4_dry.md`)

1.063 pozitif kare (05: 473, 11: 590). Satır 6 = tek kare hacim tepesi her karede; satır 7 = filtre.

| satır | değerlendirme | med6 → med7 | azalma [blok bootstrap %95] | Wilcoxon p | başarı@25 | başarı@50 | κ | kapsama 7 | yarıçap medyan | H4 (i) (ii) |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 | havuz (κ aynı veride) | 61,5 → 38,3 m | **0,376** [0,198; 0,823] | 2,9·10⁻¹⁹ | 0,235 → 0,298 | 0,462 → 0,608 | 3,25 | 0,900 | 173 m | ✓ ✓ |
| 7 | κ 05 → 11 | 53,7 → 34,6 m | 0,355 [0,215; 0,639] | 5,5·10⁻¹⁹ | 0,249 → 0,336 | 0,488 → 0,675 | 3,75 | 0,947 | 176 m | ✓ ✓ |
| 7 | κ 11 → 05 | 121,0 → 44,6 m | 0,632 [0,081; 0,848] | 5,5·10⁻⁴ | 0,218 → 0,252 | 0,429 → 0,524 | 2,66 | 0,789 | 285 m | ✓ ✗ |
| 7-noupd | havuz | 61,5 → 40,9 m | 0,334 | 4,4·10⁻²⁰ | 0,235 → 0,278 | 0,462 → 0,570 | 3,38 | 0,900 | 181 m | ✓ ✓ |
| 7-cv | havuz | 61,5 → 57,4 m | 0,067 [−0,123; 0,653] | 0,94 | 0,235 → 0,231 | 0,462 → 0,464 | 5,43 | 0,900 | 717 m | ✗ ✓ |

- **H4 calib'de sağlanıyor:** medyan hata %37,6 azalıyor (ölçüt %20), Wilcoxon p ≈ 10⁻¹⁹; kapsama havuzda tanım gereği 0,90.
  Bootstrap alt sınırı 0,198 eşiğe çok yakın: azalma 05'in yanlış tepelerinden geliyor, bloklar arası değişken.
- **Kazancın kaynağı drone↔drone odometrisi:** sabit hız (7-cv) neredeyse hiç kazanç vermiyor (0,067, p = 0,94).
  M3 kapısı (7 vs 7-noupd) küçük ama tutarlı katkı veriyor (0,376 vs 0,334).
- **Yarıçap:** filtre medyanı 173 m; tek kare q̂σ̂ medyanı 277 m (kabul edilenlerde 206 m) → filtre bölgesi daha dar.
- **Zayıf yanlar:** (1) yarıçap yine de geniş (173 m); (2) κ bölgeler arası taşınmıyor
  (11 → 05 kapsama 0,789), H3'teki FAR kaymasının aynısı; (3) GT gürültü tabanı (F5-2) yüzünden kabul edilmiş karelerde
  kazanç sınırlı: başarı@25 yalnız 0,235 → 0,298.

## 5C. γ eşikli uydu eşleştirmesi (`docs/sonuclar/faz5/gamma.md`, `sekiller/gamma.png`)

Tahmin adımından sonra yayılım (ağırlıklı RMS) ≤ γ ise o karede M1 → RoMa → M2 → M3 çalışmaz. Calib 05 + 11, filtre ayarı
ve κ sabit; kural sonuçtan önce (F5-6).

| γ (m) | atlanan | medyan hata | başarı@25 | başarı@50 | kapsama | kural |
|---|---|---|---|---|---|---|
| 0 | 0,000 | 38,3 m | 0,298 | 0,608 | 0,900 | ✓ |
| 25 | 0,015 | 38,5 m | 0,297 | 0,605 | 0,903 | ✓ |
| **50** | **0,296** | **39,8 m** | 0,281 | 0,593 | 0,912 | ✓ **seçilen** |
| 75 | 0,471 | 43,0 m | 0,244 | 0,575 | 0,914 | ✗ |
| 100 | 0,563 | 49,3 m | 0,222 | 0,508 | 0,916 | ✗ |
| 150 | 0,639 | 63,5 m | 0,161 | 0,409 | 0,893 | ✗ |
| 300 | 0,874 | 165,3 m | 0,091 | 0,219 | 0,754 | ✗ |

Hesap kazancı: uydu eşleştirmesi kare başına ≈ 0,48 s (4G), drone↔drone eşleştirmesi ise her karede gerekiyor. γ = 50 m ile
karelerin ≈ %30'unda uydu yolu atlanıyor, hata %4 artıyor. Daha fazlası hızla bozuyor: atlanan karede M3 kabul de yok, filtre
yalnız odometriyle ilerliyor.

## 5D. Görseller (`docs/sonuclar/faz5/sekiller/`, üretici `scripts/73_m4_figures.py`)

- `traj_05.png`, `traj_11.png`: GT, tek kare tahminleri (M3 reddi soluk), filtre yörüngesi, sıfırlamalar; bütün uçuş + 60 karelik pencere.
- `clouds_05.png`, `clouds_11.png`: üç anda parçacık bulutu (uzun M3 ret dizisi sonu, sıfırlamadan önce, iyi izleme) + uydu karosu.
- `err_calib.png`: kare başına hata (log), M3 reddi bantları ve sıfırlamalar.
- `gamma.png`: 5C ödünleşim eğrisi. `regions.png`: bölge başına medyan hata.

Görsellerden çıkan başarısızlık türü: **uzun M3 ret dizilerinde odometri sürüklenmesi.** Filtre hatası, son kabulden bu yana
geçen ret sayısıyla büyüyor (calib, satır 7):

| ardışık ret | 0 | 1–2 | 3–5 | 6–10 | 11–20 | > 20 |
|---|---|---|---|---|---|---|
| medyan hata | 30 m | 45 m | 65 m | 100 m | 154 m | 1.853 m (n = 3) |
| kare | 618 | 242 | 112 | 62 | 26 | 3 |

Adım başına ≈ %10'luk odometri kayması (5A) ret dizisi boyunca birikiyor; sonraki kabul edilen kare bulutu açıklamıyorsa
sıfırlama toparlıyor. İyileştirme yolu (yapılmadı): ret dizisinde de zayıf ağırlıklı güncelleme ya da odometri ölçeğini durum
vektörüne katmak.

## Faz 5 özeti (26–27.11.2026)

- **5A:** kareler sık (60–130 m, örtüşme 0,7–0,8); drone↔drone homografisi %99,9 kullanılır. GT (GPS) kare başına ≈ 20 m
  gürültülü (F5-2): görüntü↔uydu 13 m, görüntü↔GPS 26 m.
- **5B:** parçacık filtresi `guvenloc/m4_filter.py` (odometri tahmini, θ'ya göre toplanmış M2 hacmi P(x, y), sağlam θ karışımı,
  sensör sıfırlaması; F5-3). Ayar yalnız eğitimde (F5-4): 337 → 39 m.
- **H4 kuru denemesi (calib):** medyan hata 61,5 → 38,3 m (**−%37,6**, Wilcoxon p ≈ 3·10⁻¹⁹), kapsama 0,90 (κ = 3,25, F5-5);
  beş tohumda azalma 0,373–0,384. Kazanç odometriden (sabit hız: %6,7, anlamsız).
- **Açık riskler:** κ bölgeler arası taşınmıyor (11 → 05 kapsama 0,789); azalmanın bootstrap alt sınırı 0,198; uzun ret
  dizilerinde sürüklenme; GPS gürültü tabanı kabul edilmiş karelerdeki kazancı sınırlıyor.
- **5C:** γ = 50 m, uydu eşleştirmesinin %30'u atlanıyor, hata +%4 (F5-7).
- **Faz 6'ya:** H4 `configs/h4.yaml` (h4_v1), `tune.json`, κ = 3,25, tohum 0 ile test bölgelerinde tek sefer.

Kararlar F5-1…F5-7. Test bölgelerine (02, 03, 06) dokunulmadı.

