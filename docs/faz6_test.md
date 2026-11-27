# Faz 6 — test değerlendirmesi, ablasyon, hız, prototip, açık kaynak

Test bölgeleri 02, 03, 06 (6.525 çift, 2.177 pozitif kare). Ön-kayıtlar: `configs/h1.yaml` (h1_v3), `h2.yaml` (h2_v3),
`h3.yaml` (h3_v1), `h4.yaml` (h4_v1). Bu belge test sonuçlarından ÖNCE başlatıldı; F6-1…F6-3 testten önce commit edildi.

## Kararlar (27.11)

| # | Karar | Gerekçe |
|---|---|---|
| F6-1 | **Test protokolü: hazırla → dondur → tek sefer.** (1) `scripts/80_final_prep.py` test girdilerini etiket özetlemeden üretir (RANSAC/hacim öznitelikleri, M1-C olasılıkları, drone↔drone hareket, filtre ölçümleri); (2) `scripts/81_freeze.py` değerlendirmeyi etkileyen her dosyanın sha256'sını `configs/final_manifest.json`'a yazar, commit edilir; (3) `scripts/90_ablation.py --final` H1 (56) → H2 (61) → H3 (64) → H4 (72) → ablasyonu sırayla koşar. Her adım `guvenloc/final_guard` ile manifesti doğrular ve kilit bırakır; ikinci koşu reddedilir. Kilidi elle silmek ancak kod hatası için, gerekçesi buraya karar olarak yazılarak | MIMARI 6A: "final_eval yalnızca bu fazda ve tek sefer, dondurulmuş config ile". Kilit, testi görüp ayar değiştirip yeniden koşmayı mekanik olarak engeller |
| F6-2 | **Test kodu calib'de prova edildi ve birebir doğrulandı (testten önce).** Teste genişletilen betikler calib'de eski sonuçları aynen üretiyor: 56 (H1) karar dosyası bayt bayt aynı; 61 (H2) 117 alanın hepsi aynı; 64 (H3) `--rehearse` 05+11 → 05+11 satır 6 aynı; 72 (H4) `--rehearse` satır 7 ve 7-γ Faz 5 calib ile aynı. Testte eşikler/parametreler calib ve eğitimden gelir, testte hiçbir şey seçilmez: satır 1–3 τ eğitim, 4e q val 04, 5a ve 2v τ calib'de 4a başarısına eşlenmiş (h2), M3 τ/q̂ calib (F4-7), M4 ayar eğitim, κ ve γ calib (F5-5, F5-7) | Testte tek koşu hakkı olduğundan kod yolu önceden doğrulanmalıydı |
| F6-3 | **Ablasyon tanımları (testten önce):** satır 4 = 4e (H1 satırı); satır 5 = M1-C 4a dönmesi, konum hacim tepesi, skor S, eşik h2 calib eşlemesi; satır 6 = aynı konum, kabul S > τ (conformal), yarıçap q̂σ̂; satır 7 = filtre, yalnız pozitif kareler (negatif çift kavramı yok). Başarı@k = kabul ∧ hata ≤ k; medyan hata tüm pozitiflerde kabulden bağımsız; AUROC skor ~ (pozitif ∧ hata ≤ 25 m, satırın kendi konumu). Ardışık satırlar 2–1, 3–2, 4–2, 5–4, 6–5, 7–6: McNemar + Wilcoxon, 12 test birlikte Holm. Kırılımlar: arazi, irtifa (eğitim+calib AGL üçte birlik sınırları), yön (en yakın ana yön), bölge | Satır tanımları MIMARI 6A'yı izler; H2'nin ortak etiketi (RANSAC merkezi) ret skorunu sınar, ablasyonun satır 5'i ise M2'nin konumunu kullanır (ikisi ayrı raporlanır) |
| F6-4 | **Test bir kez koşuldu (27.11, `scripts/90_ablation.py --final`), kilitler yerinde; sonuçlar olduğu gibi raporlanır.** Koşudan sonra M1-C'nin test yön hatası (82,4°) calib'e (8,9°) göre şüpheli bulundu ve kod yolu ayrıca denetlendi: test olasılıkları bağımsız canlı yoldan (`67.M1Live`, 4G'de calib'de %100 uyuşan) 12 çiftte yeniden hesaplandı, sonuçlar birebir aynı → kod hatası yok, yeniden koşu yok | F6-1: kilit yalnız kod hatasında açılır; hata bulunmadı |

## 6A. Test sonuçları (`docs/sonuclar/faz6/`)

### Hipotezler (ön-kayıtlı ölçütler, test 02 + 03 + 06)

| | ölçüt | test | calib (kuru deneme) | karar |
|---|---|---|---|---|
| **H1** (i) | M1-C medyan yön hatası ≤ 10° | **82,4°** | 8,9° | ✗ |
| H1 (ii) | çağrı/pozitif kare ≤ 1,5 | 1,59 | 1,38 | ✗ |
| H1 (iii) | başarı@25 farkı (4e − 2) GA alt sınırı > −0,03 | −0,022 [−0,033; −0,011] | −0,054 [−0,072; −0,036] | ✗ (kıl payı) |
| **H2** (i) | yanlış kabul 5a < 4a, McNemar p < 0,05 | 5a daha çok yanlış kabul ediyor (p = 8·10⁻⁵¹, ters yönde) | aynı | ✗ |
| H2 (ii) | AUROC(5a) − AUROC(4a) GA alt sınırı > 0 | +0,003 [−0,002; +0,007] | −0,008 [−0,016; +0,001] | ✗ |
| **H3** (i) | FAR-neg ≤ 0,05 | **0,088** [0,080; 0,097] | 0,049 | ✗ |
| H3 (ii) | kapsama ≥ 0,88 | 0,875 [0,855; 0,894] | 0,903 | ✗ (kıl payı) |
| H3 (iii) | zor çeyrek kapsaması ≥ 0,88 | 0,669 | 0,774 | ✗ |
| **H4** (i) | medyan hata azalması ≥ %20, Wilcoxon p < 0,05 | **%86,5** (207,6 → 28,1 m), p = 3·10⁻¹⁵ | %37,6 | ✓ |
| H4 (ii) | filtre kapsaması ≥ 0,88 | **0,906** | 0,900 | ✓ |

**H4 sağlandı; H1, H2, H3 sağlanmadı.** Ölçütler testten sonra değiştirilmedi.

### Yedi satırlık ablasyon (`ablation_test.md`)

| # | yöntem | başarı@25 [GA] | başarı@50 | medyan hata | AUROC | FAR-neg | çağrı/kare | kapsama | ms/kare |
|---|---|---|---|---|---|---|---|---|---|
| 1 | RoMa v2 (0°) + RANSAC | 0,352 [0,268; 0,434] | 0,443 | 298 m | 0,975 | 0,078 | 1,00 | – | 367 |
| 2 | 4 yön + en çok inlier | 0,399 [0,311; 0,486] | 0,505 | 37 m | 0,962 | 0,082 | 4,00 | – | 1.467 |
| 3 | 4 yön + en düşük entropi | 0,289 [0,213; 0,365] | 0,373 | 405 m | 0,977 | 0,076 | 4,00 | – | 1.467 |
| 4 | + M1-C (4e) | 0,377 [0,288; 0,464] | 0,477 | 120 m | 0,968 | 0,079 | 2,05 | – | 796 |
| 5 | + M2 (hacim tepesi, S) | 0,352 [0,268; 0,437] | 0,456 | 208 m | 0,975 | 0,142 | 1,00 | – | 462 |
| 6 | + M3 (conformal) | 0,352 [0,267; 0,435] | 0,455 | 208 m | 0,975 | 0,088 | 1,00 | 0,875 | 463 |
| 7 | + M4 (filtre) | **0,451** [0,361; 0,539] | **0,617** | **28 m** | – | – | 1,00 | 0,906 | 684 (γ ile 547) |

Ardışık karşılaştırmalar (Holm): 2 > 1, 2 > 3, 2 > 4, 4 > 5 anlamlı (p < 10⁻⁵); 6 = 5 (aynı konum, yalnız eşik); **7 > 6**
(McNemar yalnız 7 doğru 253 / yalnız 6 doğru 38, p = 3·10⁻³⁹; medyan hata 208 → 28 m, p = 2·10⁻¹⁴).

### Bölgeler ve yorum

| bölge (arazi) | satır 2 başarı@25 · medyan | satır 6 | satır 7 | M1-C yön hatası | H3 FAR-neg / kapsama | H4 kapsama |
|---|---|---|---|---|---|---|
| 02 (tarım) | 0,168 · 500 m | 0,130 · 381 m | 0,232 · 146 m | 26,5° | 0,078 / 0,739 | 0,889 |
| 03 (kent) | 0,642 · 20 m | 0,585 · 22 m | 0,716 · 19 m | 122,0° | 0,079 / 0,976 | 0,979 |
| 06 (dağlık) | 0,580 · 21 m | 0,527 · 24 m | 0,541 · 23 m | 101,5° | 0,143 / 0,785 | 0,793 |

- **M1-C (H1) test bölgelerine genellemedi.** 02'de medyan 26,5°, 03 ve 06'da ≈ 100–120° (çok sayıda 180° ters
  dönme). Eğitim bölgelerinde hem kentsel arazi hem çapraz uçuş yönleri vardı (pozitif karelerin %41'i), yani bu iki
  açıklama tek başına yetmiyor; neden açık. Buna rağmen satır 4'ün başarısı satır 2'ye kıl payı yakın (−0,022):
  RoMa v2 dönmeye büyük ölçüde dayanıklı (satır 1 = 0° bile 03 ve 06'da 0,54–0,58), M1'in yanlış dönmesi
  çoğu zaman eşleşmeyi bozmuyor. Kâhin dönme (4k) de satır 2'ye ulaşmıyor (−0,027): Faz 2'deki tavan bulgusu testte de geçerli.
- **H2:** hacim skoru S, inlier sayısına göre ayırt edicilikte eşit (AUROC farkı +0,003), ama calib'de eşlenen eşik
  testte daha çok yanlış kabul ediyor (FAR-neg 0,142 vs 0,079): eşik bölgeler arası taşınmıyor.
- **H3 (beklenen risk gerçekleşti):** FAR-neg 0,088 (06'da 0,143). Faz 4'te kuru denemelerde görülen bölge kayması
  (negatiflerin sahte eşleşme düzeyi bölgeye bağlı) testte de var; tuzak normalizasyonu bunu düzeltiyordu ama AUROC
  bedeli yüzünden seçilmemişti (F4-6). Kapsama 0,875 (ölçüt 0,88), zor çeyrek 0,669.
- **H4:** filtre tek kareye göre medyan hatayı 208 → 28 m indiriyor; kazanç büyük ölçüde 02'den (381 → 146 m), 03 ve
  06'da tek kare zaten iyi (22–24 m), filtre az iyileştiriyor (19 ve 23 m). Sabit hız modeli (7-cv) yine başarısız
  (02'de 742 m); odometri şart. Bootstrap alt sınırı 0,193 (havuzlanmış azalmanın %95 GA'sı [0,193; 0,898]), yani
  ölçüt nokta tahminiyle sağlanıyor ama belirsizlik geniş. Kapsama 06'da 0,793 (κ taşınması, Faz 5 riski).
- **GT gürültü tabanı (F5-2)** 03 ve 06'daki medyanları (19–24 m) sınırlıyor; bu bölgelerde hata zaten GPS gürültüsü mertebesinde.

## 6B. Hız ve bellek (`docs/sonuclar/faz6/speed.md`, `scripts/91_speed.py`)

RTX 3090, RoMa v2 `base`, 60 calib karesi, bileşen medyanları: RoMa 260 ms, uydu karosu kırpma 95 ms, M1-C (3 ağ ×
4 TTA) 43 ms, hacim 53 ms, RANSAC 12 ms (CPU), drone↔drone RoMa 221 ms, filtre adımı 0,4 ms. Satır 7 ≈ 684 ms/kare
(γ = 50 m ile 547). **En yüksek GPU belleği 3,59 GB → 8 GB'lık kartta çalışır.** M1'in çağrı tasarrufu: satır 4'te
2,05 çağrı/kare (satır 2/3: 4) → kare başına ≈ 670 ms kazanç; ama testte bu tasarruf başarı@25'te −0,022 bedelle geliyor.

