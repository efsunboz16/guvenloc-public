# Faz 4 — Modül 3: conformal kalibrasyon (İP4)

> Dal `faz4/m3`. Test bölgeleri (02, 03, 06) kapalı; H3 kararı Faz 6'da, `configs/h3.yaml` ile tek sefer.

## Kararlar (26.11)

| # | Karar | Gerekçe |
|---|---|---|
| F4-1 | **H3 ön-kaydı** `configs/h3.yaml` (h3_v1, sonuçlardan önce, `0015cfd`). Satır 6 = 5a (h2_v3 S, M1-C birleşimi 4a dönmesi, hacim tepesi) + split conformal; σ̂ = hacim tepe yayılımı `spread_m` (≥ 2 m); α = 0,05, β = 0,10; kalibrasyon 05 + 11 birlikte. Ölçütler testte: FAR-neg ≤ 0,05, kabul koşullu kapsama ≥ 0,88 genelde ve σ̂'nin en büyük çeyreğinde. İkincil satırlar: 6-std (sabit yarıçap), 6-M (Mondrian σ̂ dilimleri, B planı), 6-inl (inlier skoru) | MIMARI 4B–4F'nin işlemsel tanımı. 6-std "uyarlamalı"nın katkısını, 6-M B planını, 6-inl M2'siz eşdeğeri gösterir |
| F4-2 | Kabul kesin büyüklükle (S > τ) | Sürekli skorlarda plandaki S ≥ τ ile aynı; eşitlikte garantiyi korur |
| F4-4 | **Tuzak karo normalizasyonu, benzetim** (`scripts/65_m3_decoy.py`): kare başına sahte eşleşme düzeyi bilinen yanlış bir karoyla ölçülür; S_norm = S − S(tuzak), inlier_norm = log1p(n) − log1p(n_tuzak). Benzetimde tuzak = komşu karenin easy0 çifti (karenin üç çiftinde de aynı). Seçim yalnız eğitim bölgelerinde, bölge-dışarıda; kural sonuçtan önce yazıldı: ort. \|FAR − α\| ve en büyük FAR birlikte düşmeli, AUROC kaybı ≤ 0,005. **Sonuç: kuralı geçen yok → ham S kalır** | F4-3'teki FAR kayması (05 ≫ 11) eğitim bölgeleri arasında da var: ham S'de FAR 0,012–0,243 |
| F4-5 | **Gerçek tuzak çiftleri** (`scripts/66_m3_decoy_pairs.py`, `cache/decoy_v1`): her kare × başka bölgeden rastgele karo (komşu karenin easy0 konumu), aynı karede, 4 dönme; train + calib (test Faz 6'da). Aynı aileler, aynı seçim kuralı (F4-4), yalnız tuzak kaynağı değişti. Sonuçtan önce commit edildi | Benzetimde AUROC kaybının olası kaynağı tuzağın başka kareden gelmesi (karenin kendi sahte eşleşme düzeyini değil komşusununkini ölçüyor). Dağıtımdaki kurulum zaten aynı kare; maliyet +1 RoMa çağrısı/kare (≈ 257 ms) |
| F4-6 | **Tuzak hattı kapandı, ham S kalır** (F4-5 sonucu, kural değiştirilmeden): gerçek tuzakta da kuralı geçen yok. h3_v1 ön-kaydı olduğu gibi Faz 6'ya gider; tuzak normalizasyonu ikincil bulgu olarak raporlanır, H3 ölçütleri ve skor değişmez | S_norm FAR'ı yine düzeltiyor (ort. sapma 0,086 → 0,026, en büyük 0,243 → 0,097) ama AUROC 0,959 → 0,926 ve doğru pozitif kabulü 0,991 → 0,724. Benzetimle neredeyse aynı: AUROC kaybı tuzağın kaynağından değil, normalizasyonun kendisinden (tuzak skoru gürültüsü farka ekleniyor). Testte (02, 03, 06) FAR-neg > 0,05 riski açık kalır |
| F4-7 | **4G hat parametreleri sabitlendi** (`scripts/67_m3_pipeline.py --export` → `docs/sonuclar/faz4/m3_params_h3_v1.json`): S = h2_v3 lojistik (eğitim), τ = −4,761 (n⁻ 2035), q̂ = 20,67 (n⁺ kabul 618), 05+11 calib, satır 6. Faz 6 testi bu dosyayla, yeniden kalibrasyon yok | h3_v1 ön-kaydının (F4-1) doğrudan uygulaması; entropi NaN dolgusu eğitim en büyüğü (calib'de NaN yok) |
| F4-3 | Kuru denemeler (karar değil): 05 → 11, 11 → 05, 05+11 → 04 (S 04 dışarıda bırakılarak yeniden eğitilir; 04 M1-C eğitiminde yok) | Test açılmadan bölge kaymasının büyüklüğünü görmek (MIMARI risk tablosu, Faz 4) |

## 4A. Sentetik doğrulama — `guvenloc/m3_conformal.py`, `tests/test_m3_coverage.py`

200 tekrar × (10.000 kalibrasyon + 10.000 test), heteroskedastik 2B Gauss hata, σ̂ = σ · log-normal gürültü:
FAR-neg ortalaması α ± 0,002, kapsama 1 − β ± 0,003; Mondrian'da zor çeyrek ≥ 0,89; sabit yarıçaplı standart
conformal genelde tutuyor ama zor çeyrekte < 0,80 (testle gösterildi). 5 test, 1,3 s.

## 4B–4F. Kuru denemeler (`docs/sonuclar/faz4/h3_dry.md`)

Satır 6 (ana), FAR-neg / kapsama / zor çeyrek / yarıçap medyanı:

| kalibre → değerlendir | FAR-neg | kapsama | zor çeyrek | yarıçap | H3 ölçütleri |
|---|---|---|---|---|---|
| 05 → 11 | 0,003 | 0,997 | 1,000 | 185 m | ✓ ✓ ✓ |
| 11 → 05 | **0,227** | 0,673 | 0,415 | 88 m | ✗ ✗ ✗ |
| **05+11 → 04** (kent; ön-kayıttaki kurulum) | 0,024 | 0,977 | 0,907 | 115 m | ✓ ✓ ✓ |
| 05+11 → 05+11 (örneklem içi, denetim) | 0,049 | 0,903 | 0,774 | 206 m | ✓ ✓ ✗ |

İkincil satırlar (05+11 → 04): 6-std ✓ ✓ ✓ (yarıçap 304 m), 6-M ✓ ✗ ✗ (kapsama 0,73), 6-inl ✓ ✗ ✗ (0,62).

**Teşhis:**
1. **FAR-neg bölgeler arası taşınmıyor.** 05'in negatifleri çok daha fazla sahte inlier topluyor (95. yüzdelik
   1.629 / 11'de 661; S medyanı −8,4 / −10,7). 05 daha alçak (AGL medyanı 509 / 787 m) ve dokusu tekrarlı.
   Tek bölgeyle kalibre edilen τ öbür bölgede ya çok tutucu (05 → 11) ya çok gevşek (11 → 05). İki bölge
   birlikte kullanıldığında τ'yu "zor" bölge (05) belirliyor; 04'te FAR-neg 0,024. Testte 05'ten daha zor bir
   bölge varsa FAR-neg %5'i aşar. **H3'ün asıl riski bu.**
2. **Yarıçaplar büyük (medyan 100–200 m).** Kabul edilen pozitiflerin yaklaşık yarısı 25 m dışında
   (S doğru karoyu buluyor ama konumun doğru olduğunu ayırt edemiyor; FAR-loc). Hacim yayılımı σ̂
   (medyan 15 m) yanlış konumlu tepeyi görmüyor; %90 kapsama için q̂ bu kareleri de kapsamak zorunda.
3. **Zor çeyrek σ̂ ile iyi tanımlanmıyor:** örneklem içinde bile 0,77. Mondrian (6-M) örneklem içinde düzeltiyor
   (0,91), ama 04'e taşınmıyor (0,73): yanlış konum oranı bölgeye göre değişiyor.

**Durum:** M3 kodu ve H3 ön-kaydı tamam; ön-kayıttaki kurulum (05+11) tek bağımsız bölgede (04) üç ölçütü de
sağlıyor. Testte iki risk var: FAR-neg'in daha zor bir bölgeye taşınmaması ve "zor çeyrek" kapsaması.

## F4-4. Tuzak normalizasyonu — benzetim (`docs/sonuclar/faz4/decoy_sel.md`)

Eğitim bölgeleri, bölge-dışarıda (S ve τ dışarıda bırakılan bölge olmadan), satır 2 dönmesi:

| skor | FAR 01 / 04 / 08 / 09 / 10 | ort. \|FAR − α\| | en büyük FAR | AUROC | doğru poz. kabul |
|---|---|---|---|---|---|
| inlier | 0,161 / 0,004 / 0,061 / 0,005 / 0,111 | 0,055 | 0,161 | 0,952 | 0,914 |
| inlier_norm | 0,124 / 0,014 / 0,047 / 0,018 / 0,160 | 0,051 | 0,160 | 0,822 | 0,616 |
| S | 0,184 / 0,012 / 0,019 / 0,016 / 0,243 | 0,086 | 0,243 | **0,959** | 0,991 |
| S_norm | 0,097 / 0,045 / 0,042 / 0,025 / 0,035 | **0,020** | **0,097** | 0,927 | 0,717 |

S_norm FAR taşınmasını belirgin biçimde düzeltiyor, ama AUROC'u 0,03 düşürüyor (kural ≤ 0,005) → geçmiyor.

## F4-5. Tuzak normalizasyonu — gerçek tuzak çiftleri (`docs/sonuclar/faz4/decoy_sel_real.md`)

Aynı kurulum (eğitim bölgeleri, bölge-dışarıda, satır 2 dönmesi); tuzak = aynı kare × başka bölgeden karo, 10.350 çift:

| skor | FAR 01 / 04 / 08 / 09 / 10 | ort. \|FAR − α\| | en büyük FAR | AUROC | doğru poz. kabul |
|---|---|---|---|---|---|
| inlier | 0,161 / 0,004 / 0,061 / 0,005 / 0,111 | 0,055 | 0,161 | 0,952 | 0,914 |
| inlier_norm | 0,130 / 0,013 / 0,048 / 0,016 / 0,132 | 0,047 | 0,132 | 0,824 | 0,635 |
| S | 0,184 / 0,012 / 0,019 / 0,016 / 0,243 | 0,086 | 0,243 | **0,959** | 0,991 |
| S_norm | 0,097 / 0,046 / 0,031 / 0,022 / 0,083 | **0,026** | **0,097** | 0,926 | 0,724 |

İki normalize skor da FAR'ı düşürüyor, ama AUROC kaybı 0,03 (S) ve 0,13 (inlier); kural (≤ 0,005) geçmiyor → **ham S kalır (F4-6)**.
Benzetim ile gerçek tuzak sonuçları birbirine çok yakın: düzeltme etkisi ve ayırt edicilik kaybı tuzağın nereden geldiğine
bağlı değil. Ödünleşim açık: tuzak, bölgeler arası FAR taşınmasını yaklaşık üçte bire indiriyor, karşılığında doğru konumların
yaklaşık %27'si reddediliyor.

## 4G. Uçtan uca hat — `guvenloc/pipeline.py` (`docs/sonuclar/faz4/e2e.md`)

`Localizer.localize(frame, tile, scale_s, tile_size_m, rot_probs) -> Decision{xy, latlon, theta, sigma_hat, score, accept,
radius_m, rot, n_inliers}`. Akış: M1-C birleşimi (+TTA, `56.probs_arrays`) → en olası dönme → RoMa v2 (tek çağrı) →
RANSAC + oylama hacmi → S → conformal. M1-C ağı pakete taşınmadı (eğitim betiklerine bağlı); `rot_probs` dışarıdan verilir.

Denetim 1 (RoMa'sız): çevrimdışı calib öznitelik satırları → `Localizer.decide`: en büyük |ΔS| = 6·10⁻¹⁵, kabul ve yarıçap
%100 aynı. Denetim 2 (canlı, 100 kare = 05 ve 11'den eşit aralıklı, 289 çift; değerlendirme değil, τ/q̂ aynı veriden):

| ölçü | değer |
|---|---|
| dönme uyuşması (canlı M1-C vs 4a) | 1,000 |
| kabul uyuşması | 1,000 |
| \|ΔS\| medyan / %95 | 0,0000 / 0,0003 |
| hacim tepesi kayması | 0,00 m |
| FAR-neg canlı / çevrimdışı | 0,037 / 0,037 |
| pozitif kabul / kapsama | 0,490 / 0,898 |
| konum hatası medyan (kabul ∧ pozitif) | 33,0 m |
| süre / çift (karo + M1 + RoMa/hacim) | 0,48 s (0,14 + 0,04 + 0,29) |

Canlı hat önbellekli hatla aynı kararı veriyor (RoMa `seed` ile belirlenimci); enlem/boylam çıktısı piksel hatasıyla 0,6 m
içinde. Faz 4'ün "bitti" ölçütü (H3) test bölgelerinde → Faz 6.

## Faz 4 özeti (kapanış 26.11.2026, `v0.3`)

- **Split conformal (4A):** `guvenloc/m3_conformal.py` (τ, q̂, Mondrian σ̂ dilimleri). Sentetik testte FAR-neg ve
  kabul koşullu kapsama hedefte (200 × 10.000; `tests/test_m3_coverage.py`).
- **H3 ön-kaydı (F4-1):** `configs/h3.yaml` h3_v1, satır 6 = h2_v3 skoru S + σ̂ = spread_m, kalibrasyon 05+11.
  Ölçütler (FAR-neg ≤ 0,05; kapsama genel ve zor çeyrekte ≥ 0,88) yalnız test bölgelerinde, Faz 6'da.
- **Kuru denemeler (4B–4F):** 05+11 → 04 üç ölçütte de ✓. Tek bölgeden kalibrasyonda FAR taşınmıyor
  (11 → 05 FAR-neg 0,227): negatiflerin sahte inlier düzeyi bölgeye bağlı.
- **Tuzak normalizasyonu (F4-4, F4-5):** FAR kaymasını yaklaşık üçte bire indiriyor (ort. sapma 0,086 → 0,026),
  ama AUROC −0,033 ve doğru konumların ≈ %27'si reddediliyor. Önceden yazılan kural geçmedi; ham S kaldı (F4-6).
- **Uçtan uca hat (4G, F4-7):** `guvenloc/pipeline.py`. 100 karelik canlı test çevrimdışı hatla birebir
  (dönme ve kabul uyuşması 1,000), 0,48 s/çift.
- **Açık risk:** test bölgelerinde FAR-neg > 0,05 olabilir (eğitim bölgelerinde bölge-dışarıda ham S ile
  FAR 0,012–0,243). Faz 6'da h3_v1 ve `m3_params_h3_v1.json` ile aynen koşulacak; sonuç ne olursa raporlanır.
- **Sonraki faza:** M4 (zamansal füzyon) tek kare kararını `Decision` olarak alır; σ̂ ve yarıçap füzyonun girdisi.

Kararlar F4-1…F4-7. Test bölgelerine (02, 03, 06) dokunulmadı.

