# Faz 1 — kalan işler (üniversite sunucusu, RTX 3090, 25 Kasım 2026)

> Bu belge Faz 1'i sunucuda bitirmek için gereken her şeyi sırasıyla anlatır. 25.11'den itibaren bütün GPU
> işleri bu sunucuda yapılır (MIMARI "Çalışma ortamı").
> Durum özeti ve kararların gerekçeleri: [faz1_durum.md](faz1_durum.md).

## 0. Ne bitti, ne kaldı?

**Bitti (dizüstünde):** veri doğrulama, coğrafi altyapı, ayrım ve çift listesi, eşleştirici + önbellek
kodu, metrik/istatistik kodu, WxBS tekrarlaması (55,91 ✅), ilk taban çizgisi tablosu, karo deneyi (K2).

**Verilmiş kararlar (artık tartışma yok):**
- **K2 — karo:** uyarlamalı (ayak izi köşegeni + 2 × 200 m). `configs/pairs.yaml: tile_mode: adaptive`.
  `pairs_v1.csv` buna göre yeniden üretildi (19.973 çift, checksum `docs/checksums/pairs_v1.sha256`).
- **K6 — önbellek ayarı:** `base` (`configs/base.yaml: romav2.setting_cache: base`). mega1500 yalnızca
  MegaDepth tekrarlamasında.

**Kalan:**

| # | İş | Neden | Nasıl |
|---|---|---|---|
| 1 | MegaDepth-1500 + C4 tekrarlaması | RoMa v2 kurulumumuzun makaleyle aynı sonucu verdiğini göstermek | betik (§2) |
| 2 | Tam önbellek `cache/v1` (≈ 86.600 dosya) | Faz 2–4'teki bütün modüller RoMa'yı yeniden koşmadan bu dosyaları okur | betik (§2) |
| 3 | Taban çizgisi tablosunun yeni önbellekle yenilenmesi | Raporun "satır 1–3" karşılaştırması | betik (§2) |
| 4 | K1 eğim analizi | GT kaymasını düzeltmenin işe yarayıp yaramadığı (önceden konmuş kural) | Claude ile yazılacak (§4) |
| 5 | `sigma_scale` kalibrasyonu | RoMa'nın belirsizlik (Σ) ölçeğini gerçek verimize ayarlamak; Modül 2'den önce şart | Claude ile yazılacak (§4) |
| 6 | Kapanış: belgeler, PR #2 → main, `v0.0` etiketi, danışman özeti | Faz 1'in resmi bitişi | §5 |

## 1. Hazırlık

Sunucu (Ubuntu 22.04, RTX 3090) — 25.11'de yapıldı:
- [x] Depo klonlandı, `faz1/altyapi` dalı; `.venv` (Python 3.12, torch 2.11.0+cu128) kuruldu
- [x] `00_check_env.py` (RTX 3090, sm_86), `01_roma_single_pair.py --setting mega1500` (köşe hatası 0,32 px),
      `pytest` (70 geçti); RoMa v2 ağırlıkları indirildi
- [x] Uyku modu kapalı (GNOME: `sleep-inactive-ac-type = nothing`)
- [x] UAV-VisLoc paketini (dizüstündeki `data/UAV_VisLoc_dataset/` klasörünün tamamı) sunucuda
      `data/UAV_VisLoc_dataset/` altına koyun. MegaDepth'i betik kendisi indirir. Veri sunucuda kalır.

Sıfırdan kurulum gerekirse:
```bash
git clone https://github.com/efsunboz16/guvenloc.git && cd guvenloc
git checkout faz1/altyapi            # PR #2 birleşince: main
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
Uzun koşuları `tmux` içinde başlatın (`tmux new -s faz1`; ayrılmak için Ctrl+B, D; dönmek için
`tmux attach -t faz1`): oturum kapansa da koşu devam eder.

## 2. Tek komut (≈ 20 sa, gözetimsiz)

```
python scripts/faz1_3090.py --list     # adımları göster
python scripts/faz1_3090.py            # hepsini sırayla çalıştır
```

Canlı izleme (ikinci terminalde; yalnızca okur): `python scripts/faz1_izle.py` — adım durumları, önbellek
ilerleme çubukları ve kalan süre, tekrarlama tohumları, GPU/disk, son günlük satırları.

| Adım | Ne yapar | Süre (tahmin) | Doğru gitti mi? |
|---|---|---|---|
| `env` | GPU kontrolü + testler | 5 dk | `RTX 3090 … sm_86`, testler yeşil |
| `data` | Veri sayımı + checksum | 10 dk | 6.774 kare, 11 harita; checksum farkı yok |
| `pairs` | Çift listesi (uyarlamalı karo) | 5 dk | `pairs_v1.csv checksum doğru` |
| `speed` | Hız ölçümü | 5 dk | `results/faz1/speed.json` (aşağıdaki süreleri bununla düzeltin) |
| `megadepth` | MegaDepth-1500 test verisi | 10–30 dk | `data/megadepth/` altında 5 `.npz` + `Undistorted_SfM` |
| `smoke` | 25 çiftlik deneme, orijinalle karşılaştırma | 5 dk | `birebir aynı: True` (yoksa betik durur) |
| `cache_dev` | Önbellek: calib + train (≈ 54 bin dosya) | ≈ 4–5 sa | `bitti: … yeni` + doğrulamada `bozuk 0` |
| `baselines` | Taban çizgisi tablosu (CPU) | ≈ 30 dk | `results/faz1/baselines/table_v1_calib.md` |
| `repro` | MegaDepth-1500 ve C4, 5'er tohum | ≈ 10–15 sa | hedefler aşağıda |
| `cache_rest` | Önbellek: ardışık kareler + test (≈ 33 bin dosya) | ≈ 3 sa | `bozuk 0`, şema testi yeşil |

Sıra bilinçli: ekibin Faz 2'de ihtiyaç duyduğu önbellek (calib + train) önce biter.

**Kesinti olursa** (elektrik, güncelleme, Ctrl+C): aynı komutu tekrar çalıştırın ya da kalınan adımdan
`python scripts/faz1_3090.py --from cache_dev`. Önbellek var olan dosyaları, tekrarlama biten tohumları
atlar; yarım kalmış dosyalar otomatik silinir. Hata olursa betik durur ve hangi adımda kaldığını yazar;
günlük: `logs/faz1_3090.log`.

**MegaDepth hedefleri** (makale; ±1 puan içinde olmalı):
- MegaDepth-1500 AUC@5/10/20 = **62,8 / 77,0 / 86,6** → `results/faz1/repro_mega1500.json`
- C4 AUC@5 ≈ **53,45** (yön kırılımı da yazılır) → `results/faz1/repro_c4.json`
- Sapma > 1 puan ise: önce `smoke` sonucunu Kaggle T4'teki sonuçla (57,63 / 73,19 / 84,11, tohum 0)
  karşılaştırın; aynıysa fark donanımdan değil → MIMARI 1D B planı (çözünürlük, RANSAC eşiği).

## 3. Betik bittikten sonra: sonuçları GitHub'a koymak

```
git add results/runs.csv
git commit -m "Faz 1: 3090 koşusu (MegaDepth/C4, önbellek v1, taban çizgileri)"
git push
```
`results/` git dışında (yalnızca `runs.csv` izleniyor). Tabloları belgelere işlemek §4'te Claude ile.

## 4. Claude ile yapılacaklar (aynı makinede, betik bittikten sonra)

Claude Code'u repo klasöründe açıp şunu söyleyin: *"docs/faz1_kalan_isler.md §4'ü yap."*

**4a. Belgeler.** ✅ 26.11. `results/faz1/repro_*.json` ve `results/faz1/baselines/table_v1_calib.md` →
`docs/faz1_tekrarlama.md` (MegaDepth satırları; dev_base tablosunun yerine v1 tablosu) ve
`docs/sonuclar/faz1/`'e kopya. Kontrol listesini kapat.
K5 kararı (25.11) gereği ayrıca: bölge başına drone/uydu tarih farkı tablosu ve Faz 1 raporuna
"zor durumlar" paragrafı (eşleşmenin tamamen kaçtığı tarım/su/mevsim farkı örnekleri, K1 GT kayması).

**4b. K1 eğim analizi** ✅ 26.11 (`scripts/41_gt_tilt.py`: azalma %22,3 < %30 → GT = CSV). Önceden konan kural (faz1_durum.md K1):
- Eğitim bölgelerinde, satır 2'nin kabul ettiği ve hatası < 100 m olan pozitiflerde RoMa'nın bulduğu görüntü
  merkezi ile CSV `lat/lon` arasındaki kaymayı drone çerçevesine (ileri/yanal, Phi1 ile) çevir.
- Doğrusal model: `yanal ~ a + b·Omega + c·Kappa`, `ileri ~ …` — YALNIZCA eğitimde kur.
- calib'de medyan kayma öncesi/sonrası. **≥ %30 azalırsa** düzeltilmiş GT ikincil sonuç olarak `pairs_v2`
  (ana sonuç CSV GT'si kalır); azalmazsa konu kapanır. Sonuç faz1_durum.md K1'e yazılır.

**4c. `sigma_scale` kalibrasyonu** ✅ 26.11 (`scripts/42_sigma_scale.py --write`: s = 0,9272; eğitim %95 kapsama %93,2, calib %95,7).
- Eğitim pozitiflerinde satır 2'nin seçtiği doğru eşleşmelerde (hata < 25 m) RANSAC inlier'ları için artık
  `r_i = v_i − H(u_i)` (karo pikseli).
- Gauss NLL'yi en küçükleyen ölçek: `s² = mean(r_iᵀ Σ_B,i⁻¹ r_i) / 2` (Σ önbellekte `Sigma_B`).
- Kontrol: ölçeklenmiş %95 elipslerinin kapsaması eğitimde ≈ %95; calib'de rapor edilir.
- Sonuç `configs/base.yaml: romav2.sigma_scale` (şu an 1,0) + `results/runs.csv`.

**4d. Önbellek erişimi.** `cache/v1/` (MANIFEST.csv dahil, ≈ 20 GB) sunucuda kalır, kopyalanmaz: Faz 2–5'in
bütün GPU işleri sunucuda koşar. Doğrulama: `python scripts/31_verify_cache.py cache/v1`. Yedek almak isterseniz
harici diske kopyalanıp aynı komutla doğrulanır (isteğe bağlı).

## 5. Kapanış

- [x] `docs/faz1_tekrarlama.md` tabloları dolu (26.11; kontrol listesinde `sigma_scale`, C4 kaynağı, PR/etiket açık)
- [x] INTERFACES §1–2 onayı (K3, 25.11)
- [ ] PR #2: CI yeşil → `main`'e birleştir
- [ ] `git tag v0.0 && git push origin v0.0`
- [ ] Danışmana tek sayfalık özet. Özellikle belirtin: karo boyutu artık sabit değil, kare başına
      (MIMARI'deki kuralın kare başına uygulanması; yalnızca eğitimde, önceden konan kuralla seçildi);
      önbellek `base` ayarında (donanım kısıtı); WxBS ve MegaDepth tekrarlama sonuçları.

## Ek: elle çalıştırmak isterseniz

Betiğin yaptığı her şey tek tek de çalıştırılabilir (aynı sırayla):
```
python scripts/00_check_env.py && python -m pytest -q
python scripts/11_verify_data.py data/UAV_VisLoc_dataset --expect-frames 6774 --expect-maps 11 --checksums
python scripts/14_make_pairs.py data/UAV_VisLoc_dataset
python scripts/23_speed.py data/UAV_VisLoc_dataset --settings base mega1500
python scripts/20_repro_mega1500.py --data data/megadepth --limit 5 --check-original
python scripts/30_build_cache.py data/UAV_VisLoc_dataset --split calib
python scripts/30_build_cache.py data/UAV_VisLoc_dataset --split train
python scripts/31_verify_cache.py cache/v1 --delete-bad
python scripts/40_baselines.py feats --cache cache/v1 --split train --ransac-px 3 5 8
python scripts/40_baselines.py feats --cache cache/v1 --split calib --ransac-px 3 5 8
python scripts/40_baselines.py select --cache cache/v1
python scripts/40_baselines.py eval --cache cache/v1 --split calib
python scripts/20_repro_mega1500.py --data data/megadepth --seeds 0 1 2 3 4
python scripts/20_repro_mega1500.py --data data/megadepth --c4 --seeds 0 1 2 3 4
python scripts/30_build_cache.py data/UAV_VisLoc_dataset --seq
python scripts/30_build_cache.py data/UAV_VisLoc_dataset --split test
python scripts/31_verify_cache.py cache/v1 --delete-bad
```
Kaggle yolu (denendi, kullanılmadı): [faz1_kaggle.md](faz1_kaggle.md).
