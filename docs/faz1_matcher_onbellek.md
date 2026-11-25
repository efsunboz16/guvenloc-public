# Faz 0.4 + 1E — Hassasiyet birimi, `matcher.py`, önbellek

## 0.4 Hassasiyet birimi (çözüldü)

**Kaynak kod:** `romav2/refiner.py` hassasiyeti Cholesky parametreleriyle kurar
(`l00 = softplus(.) + eps  # this is in pixels`) ve refiner aşamaları boyunca biriktirir;
hr aşamasına geçerken (`patch_size == 4, stage == 1`) sıfırlar (`zero_out_precision`) — yani P her
aşamanın kendi ızgara pikselinde. Ağ görüntüyü her zaman ızgara boyutuna küçültülmüş görür; orijinal
görüntü boyutundan bağımsızdır.

**Deney:** `python scripts/02_precision_units.py data/UAV_VisLoc_dataset --setting {base,mega1500}`.
Eğitim bölgelerinden (01, 04, 08, 09) gerçek uydu karoları A (900×600), bilinen rastgele homografi
(±30° dönme, 0,8–1,25 ölçek, perspektif) + fotometrik bozulma ile B (700×700). Yoğun ızgarada gerçek
hata e bilinir; doğru birimde m = eᵀPe ≈ χ²(2) (medyan 1,386).

| Ayar | n | medyan \|e\| | hipotez | medyan m | medyan σ_B |
|---|---|---|---|---|---|
| base (lr 640) | 80.000 | 0,079 px | **lr = hr (640)** | 0,069 | 0,31 px |
| | | | anchor 512 | 0,044 | 0,39 px |
| | | | normalize | ≈ 0 | 98,8 px |
| mega1500 (lr 800 / hr 1024) | 48.000 | 0,063 px | lr 800 | 0,065 | 0,26 px |
| | | | **hr 1024** | **0,107** | **0,20 px** |
| | | | anchor 512 | 0,027 | 0,40 px |
| | | | normalize | ≈ 0 | 103,4 px |

Sonuç: P, **son aşamanın ızgara pikselinde** (base → 640, mega1500 → hr 1024). Normalize hipotezi
~1000 kat yanlış. Doğru birimde σ gerçek hatayla aynı mertebede ve temkinli (m medyanı 0,1 < 1,39);
corr(log σ, log |e|) = 0,35–0,42 → belirsizlik bilgilendirici. Kolay sentetik çiftte model temkinli
olduğundan gerçek drone↔uydu verisinde ölçek katsayısı `romav2.sigma_scale` (Σ ← k²Σ) eğitim
bölgelerinde NLL ile seçilecek (MIMARI §13 B planı) — şimdilik 1,0.
**Bitti sayılır (0.4):** `matcher.py` Σ'yı piksel biriminde döndürüyor ✓; sentetik çiftte σ ile gerçek
hata aynı mertebede ✓ (`tests/test_matcher.py::test_matcher_synthetic_homography`).

## 1E `matcher.py`

- `Matcher(setting, sigma_scale).match(img_drone, img_sat, rot_deg, n) -> MatchSet`.
- `rot_deg`: drone karesine saat yönünde dönme; 90'ın katları birebir, keyfi açılar genişletilmiş
  tuvalle (Faz 2'de M1'in θ̂'sı için). `u` ve `Sigma_A` orijinal kare koordinatına geri çevrilir.
- Örnekleme RoMa'nınkiyle birebir aynı (aynı tohum → aynı eşleşmeler; doğrulandı); KDE parça parça.
- Hız (RTX 5050 Laptop 8 GB): `base` ≈ 0,6–0,9 s/dosya (önceden 3,8 s: KDE 1,7 s + karo kesme 3 s
  düzeltildi), tepe bellek 3,6 GB. **`mega1500` bu kartta kullanılamaz: 45–56 s/dosya**, 7,9 GB —
  Windows paylaşımlı belleğe taşıyor. `mega1500` önbelleği sunucuda üretilmeli.
- `io_visloc.crop_square_m` yeniden yazıldı: kırpılmış ham okuma + INTER_AREA + alt-piksel afin;
  eskisine (rasterio boundless) göre 7–12 kat hızlı, kayma < 0,1 px. Eski uygulama
  `crop_square_m_rasterio` adıyla başvuru için duruyor.

## 1E Önbellek (`scripts/30_build_cache.py`)

- `cache/v1/<pair_id>_r<rot>.npz` (çift × {0, 90, 180, 270}), `cache/v1/seq/` (ardışık kareler).
- Yeniden başlatılabilir (var olan atlanır, atomik yazma), `MANIFEST.csv` (sha256 dahil).
- Boyut ≈ 250 KB/dosya → tamamı (19.778 × 4 + 6,7 bin seq) ≈ 20 GB.
- Şema testi: `tests/test_cache_schema.py` (cache/v1 varsa rastgele 100 dosya).
