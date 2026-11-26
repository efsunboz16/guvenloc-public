# Faz 5 — Modül 4: zamansal füzyon (parçacık filtresi), ablasyon satırı 7

Hedef 5 / H4: medyan konum hatası tek kareye göre ≥ %20 azalır, kapsama ≥ %88 korunur. Test bölgeleri (02, 03, 06) Faz 6'da.

## Kararlar (26.11)

| # | Karar | Gerekçe |
|---|---|---|
| F5-1 | **H4 ön-kaydı** `configs/h4.yaml` (h4_v1), filtre sonuçlarından önce. Satır 6 (tek kare, kabulden bağımsız her pozitif kare) vs satır 7 (filtre ağırlıklı ortalaması); bütün pozitif kareler, filtre ilk kabul edilen karede başlar (öncesi satır 7 = satır 6). Ölçüt: (med6 − med7)/med6 ≥ 0,20 ve Wilcoxon tek yönlü p < 0,05; satır 7 kapsaması ≥ 0,88 (yarıçap = κ · ağırlıklı %90 yayılım, κ calib'de). Hiper-parametreler yalnız eğitim bölgelerinde, κ ve γ calib'de | MIMARI'deki "medyan hata tek kareye göre" tanımsızdı: hangi kareler, kabul edilmeyenler ne olur. Tek kare satırı her karede bir konum veriyor (`Decision.xy`), filtre de; aynı kare kümesinde eşleştirilmiş karşılaştırma en yalın tanım |
| F5-2 | **GT gürültü tabanı ölçüldü; ölçüt değişmez.** Ardışık kare çiftlerinde (iki kare de uyduyla iyi eşleşmiş, n = 2.339) yer değiştirme farkları medyanı: görüntü↔uydu 13 m, görüntü↔GPS 26 m, uydu↔GPS 32 m. GPS (GT) kare başına ≈ 20 m bağımsız gürültü taşıyor (K1 bulgusuyla, 18–31 m, tutarlı) | Tek kare kabul edilen pozitiflerde medyan hata 28–31 m (calib) zaten bu tabana yakın; filtrenin kazancı ancak yanlış tepeli ve reddedilen karelerden gelebilir (tüm karelerde tek kare medyanı 05: 121 m, 11: 54 m). Raporlanacak sınırlılık |

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
