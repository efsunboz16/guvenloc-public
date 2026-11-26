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
