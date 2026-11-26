# Faz 5 — Modül 4: zamansal füzyon (parçacık filtresi), ablasyon satırı 7

Hedef 5 / H4: medyan konum hatası tek kareye göre ≥ %20 azalır, kapsama ≥ %88 korunur. Test bölgeleri (02, 03, 06) Faz 6'da.

## Kararlar (26.11)

| # | Karar | Gerekçe |
|---|---|---|
| F5-1 | **H4 ön-kaydı** `configs/h4.yaml` (h4_v1), filtre sonuçlarından önce. Satır 6 (tek kare, kabulden bağımsız her pozitif kare) vs satır 7 (filtre ağırlıklı ortalaması); bütün pozitif kareler, filtre ilk kabul edilen karede başlar (öncesi satır 7 = satır 6). Ölçüt: (med6 − med7)/med6 ≥ 0,20 ve Wilcoxon tek yönlü p < 0,05; satır 7 kapsaması ≥ 0,88 (yarıçap = κ · ağırlıklı %90 yayılım, κ calib'de). Hiper-parametreler yalnız eğitim bölgelerinde, κ ve γ calib'de | MIMARI'deki "medyan hata tek kareye göre" tanımsızdı: hangi kareler, kabul edilmeyenler ne olur. Tek kare satırı her karede bir konum veriyor (`Decision.xy`), filtre de; aynı kare kümesinde eşleştirilmiş karşılaştırma en yalın tanım |
| F5-2 | **GT gürültü tabanı ölçüldü; ölçüt değişmez.** Ardışık kare çiftlerinde (iki kare de uyduyla iyi eşleşmiş, n = 2.339) yer değiştirme farkları medyanı: görüntü↔uydu 13 m, görüntü↔GPS 26 m, uydu↔GPS 32 m. GPS (GT) kare başına ≈ 20 m bağımsız gürültü taşıyor (K1 bulgusuyla, 18–31 m, tutarlı) | Tek kare kabul edilen pozitiflerde medyan hata 28–31 m (calib) zaten bu tabana yakın; filtrenin kazancı ancak yanlış tepeli ve reddedilen karelerden gelebilir (tüm karelerde tek kare medyanı 05: 121 m, 11: 54 m). Raporlanacak sınırlılık |
| F5-3 | **Filtre tasarımında üç düzeltme (yalnız eğitim verisine bakılarak, calib'den önce):** (a) odometri kapısı: adım uzunluğu son 5 iyi adımın medyanının 0,5–2 katı dışındaysa son iyi adım kullanılır (sabit hız, B planı); (b) θ ölçümü sağlam karışım (1 − 0,3) N(θ − θ̂; σ) + 0,3; (c) sensör sıfırlaması: kabul edilmiş ölçüm bulutu açıklamıyorsa (ağırlıklı ortalama P/maxP < eşik) filtre o tek kare kararından yeniden başlar | İlk sürüm eğitimde tek kareden kötüydü (01: 30 → 155 m, 09: 427 → 3.463 m). Nedenler kare kare izlendi: kabul edilmiş yanlış tepelerin θ̂'sı 80–170° sapıyor ve yönü bozuyordu; dokusuz/tekrarlı zeminde homografi "takılıyor" (01'de adımların %3,8'i < 0,3 × GPS adımı); dönüşte kopan bulut karonun dışına düşünce ε tabanı onu geri getiremiyordu (hata 57 km'ye kadar) |
| F5-4 | **Filtre hiper-parametreleri eğitimde seçildi** (`scripts/72_m4_filter.py --tune`, 162 ayar, ölçüt havuzlanmış medyan hata): λ = 2, σ_pos = 5 m + 0,1 × adım, ε = 10⁻³, θ ölçümü σ = 5°, sıfırlama eşiği 10⁻⁴, N = 2.000, sıçrama p = 0,02 / 100 m. Eğitimde satır 6 → 7: 337 → 39 m | Bölge başına 01 30 → 27, 04 47 → 37, 08 471 → 172, 09 427 → 42, 10 401 → 322 m. 08 komşu ayarlar arasında 152–324 m oynuyor (kararsız) |

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
