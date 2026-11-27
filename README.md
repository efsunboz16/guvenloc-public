# GüvenLoc

Drone görüntülerini uydu haritalarıyla eşleştirerek konum ve yön tahmini yapan
araştırma projesi (TÜBİTAK 2209-A). Hat: M1-C kuzey kanonikleştirici → donuk RoMa v2 → M2 oylama hacmi ve ret skoru →
M3 split conformal kabul/ret ve güven dairesi → M4 parçacık filtresi.

**Test sonuçları (UAV-VisLoc 02 + 03 + 06, ön-kayıtlı, tek sefer — [docs/faz6_test.md](docs/faz6_test.md)):**
H4 ✓ filtre medyan hatayı 208 → 28 m indiriyor (−%86,5, kapsama 0,906); H1 ✗ (M1-C test yön hatası 82°), H2 ✗,
H3 ✗ (FAR-neg 0,088, kapsama 0,875). Satır 7 başarı@25 0,451, 684 ms/kare, en çok 3,6 GB GPU.

## Başlangıç

Projenin mimarisi, fazları, ekip rolleri ve çalışma kuralları
[MIMARI.md](MIMARI.md) dosyasında açıklanmıştır.

### Ekip geliştirme akışı

1. `main` dalını güncel tutun.
2. Her iş için `faz<N>/<kisa-ad>` biçiminde dal açın.
3. Değişiklikleri Pull Request üzerinden gönderin.
4. CI yeşil olmadan birleştirmeyin (tek geliştirici; ikinci kişi incelemesi aranmaz — MIMARI §3.1).

## Çalışma ortamı

Bütün işler (önbellek, tekrarlama, eğitim, değerlendirme) **üniversite sunucusunda**, makinenin başında yapılır:
RTX 3090 24 GB, Ubuntu 22.04. Veri, önbellek ve sonuçlar yalnızca orada durur.

## Kurulum (sunucu, Linux, Python 3.12)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/00_check_env.py
python scripts/01_roma_single_pair.py --setting mega1500
```

İlk çalıştırma RoMa v2 ağırlıklarını (~1 GB) ve DINOv3 kodunu `~/.cache/torch/hub` altına indirir.
Windows'ta: `py -3.12 -m venv .venv` ve `.\.venv\Scripts\Activate.ps1` (RTX 50 serisi için de cu128 şart).
Modüller arası veri biçimleri [INTERFACES.md](INTERFACES.md) dosyasındadır.

### Tek çift ölçümleri (sentetik çift)

| Ayar | çözünürlük (lr/hr) | GPU | süre/çift | en yüksek bellek | köşe hatası |
|---|---|---|---|---|---|
| mega1500 / wxbs | 800 / 1024 | **RTX 3090 (sunucu, 25.11)** | **0,54 s** | 7,1 GB | 0,32 px |
| precise | 800 / 1280 | RTX 5060 Laptop 8 GB (23.11) | 3,7 s | 10,4 GB (8 GB'ı aşar) | 0,31 px |
| mega1500 / wxbs | 800 / 1024 | RTX 5060 Laptop 8 GB | 2,1 s | 7,1 GB | 0,32 px |
| base | 640 | RTX 5060 Laptop 8 GB | 0,31 s | 3,4 GB | 0,34 px |
| fast | 512 | RTX 5060 Laptop 8 GB | 0,20 s | 2,6 GB | 0,33 px |

## Durum

- **Faz 0** tamam (ortam, RoMa v2 tek çift testi; 25.11'de sunucuda da doğrulandı). Hassasiyet birimi (0.4)
  Faz 1'de çözüldü.
- **Faz 1** (`faz1/altyapi`, PR #2): veri doğrulama, coğrafi altyapı, bölge ayrımı + çiftler, `matcher.py` +
  önbellek, metrikler/istatistik, WxBS tekrarlaması tamam. Sunucuda (25–26.11): MegaDepth-1500 makaleyle ±0,2
  içinde (62,98 / 76,96 / 86,55), C4 58,94 (hedef 53,45 Ge vd. 2026'da 576×576'da; fark çözünürlükten), tam önbellek `cache/v1` (86.620 dosya),
  taban çizgisi tablosu `cache/v1` ile (satır 2 başarı@25 0,319), K1 kapandı (GT = CSV).
  `sigma_scale` = 0,9272 (eğitim, NLL). Kalan: `v0.0`. Kılavuz: [docs/faz1_kalan_isler.md](docs/faz1_kalan_isler.md).
  **Özet, sorunlar ve bekleyen kararlar: [docs/faz1_durum.md](docs/faz1_durum.md).**
  Ayrıntı ve komutlar: [docs/faz1_tekrarlama.md](docs/faz1_tekrarlama.md),
  [docs/faz1_kalan_isler.md](docs/faz1_kalan_isler.md).

- **Faz 2** (`v0.1`): M1-C göreli dönme (C8 eşdeğer, birleşim + TTA). Ayrıntı: [docs/faz2_m1.md](docs/faz2_m1.md).
- **Faz 3** (`v0.2`): oylama hacmi v0, ret skoru S (h2_v3). Ayrıntı: [docs/faz3_m2.md](docs/faz3_m2.md).
- **Faz 4** (`v0.3`): split conformal (`guvenloc/m3_conformal.py`), H3 ön-kaydı `configs/h3.yaml`, uçtan uca hat
  `guvenloc/pipeline.py`:
  ```
  python scripts/67_m3_pipeline.py --export    # S + τ, q̂ → results/faz4/m3/m3_params_h3_v1.json
  python scripts/67_m3_pipeline.py --e2e 100   # canlı M1-C + RoMa, önbellekle tutarlılık testi
  ```
  Girdi: drone karesi (RGB), kuzey-yukarı uydu karosu, `scale_s`, karo kenarı (m), M1-C dönme olasılıkları.
  Çıktı: `Decision{xy, latlon, theta, sigma_hat, score, accept, radius_m}`. Ayrıntı: [docs/faz4_m3.md](docs/faz4_m3.md).

- **Faz 5** (`faz5/m4`): parçacık filtresi `guvenloc/m4_filter.py`, H4 ön-kaydı `configs/h4.yaml`:
  ```
  python scripts/70_m4_sequences.py && python scripts/71_m4_measure.py   # diziler, kare ölçümleri
  python scripts/72_m4_filter.py --tune --dry --gamma --traj               # ayar (eğitim), kuru deneme + γ (calib)
  python scripts/73_m4_figures.py; python scripts/m4_izle_web.py           # görseller, izleme paneli :8765
  ```
  Ayrıntı: [docs/faz5_m4.md](docs/faz5_m4.md).

- **Faz 6** (`faz6/test`): test sonuçları ve ablasyon [docs/faz6_test.md](docs/faz6_test.md); prototip:
  ```
  python app/app.py          # Gradio, http://127.0.0.1:7860 (Mod 1 tek kare, Mod 2 uçuş)
  ```

Veri, önbellek ve sonuç dosyaları `.gitignore` ile dışarıda tutulur (`results/runs.csv` hariç); sonuç
kopyaları `docs/sonuclar/` altındadır.

## Veri ve sürüm paketi

1. **UAV-VisLoc** (16,4 GB, Google Drive sürümü): `bash scripts/10_download_data.sh` yönergesiyle
   `data/UAV_VisLoc_dataset/` altına; doğrulama `python scripts/11_verify_data.py data/UAV_VisLoc_dataset --expect-frames 6774 --expect-maps 11 --checksums`.
2. **Sürüm paketi** `guvenloc-v1.0-paket.tar.gz` (GitHub Release eki, ≈ 145 MB; görüntü içermez): M1-C ağırlıkları,
   çift listesi, M1 etiketleri, M3 parametreleri. `python scripts/95_release.py --unpack guvenloc-v1.0-paket.tar.gz`
   sha256'ları doğrular ve dosyaları yerlerine koyar. Paket bu depodan `python scripts/95_release.py` ile üretilir.

## Tabloları yeniden üretme

Sıra önemlidir (her adım öncekinin çıktısını okur). Ağır adımların süresi RTX 3090'da verilmiştir.
Paketi açtıysanız M1 eğitimini (Faz 2) atlayabilirsiniz; çift listesi de pakettekiyle aynıdır.

| Faz | Komut | Çıktı / tablo |
|---|---|---|
| 1 veri | `python scripts/14_make_pairs.py data/UAV_VisLoc_dataset` | `data/pairs/pairs_v1.csv` (sha256 `docs/checksums/`) |
| 1 tekrarlama | `python scripts/20_repro_mega1500.py --data data/megadepth --seeds 0 1 2 3 4` (`--c4` ile C4); `python scripts/21_repro_wxbs.py --seeds 0 1 2 3 4` | [docs/faz1_tekrarlama.md](docs/faz1_tekrarlama.md) |
| 1 önbellek | `python scripts/30_build_cache.py data/UAV_VisLoc_dataset --split {train,calib,test}` ve `--seq`; `python scripts/31_verify_cache.py cache/v1` (≈ 10 saat) | `cache/v1` (RoMa v2 eşleşmeleri) |
| 1 taban çizgisi | `python scripts/40_baselines.py feats --cache cache/v1 --split train --ransac-px 3 5 8` (ve calib), `select`, `eval --split calib`; `python scripts/42_sigma_scale.py --cache cache/v1` | satır 1–3, `configs/baselines.yaml` |
| 2 M1 | `python scripts/50_m1_labels.py --cache cache/v1`; `51_m1_prep.py`; `54_m1c_prep.py --gercek --ek 3`; `bash scripts/55_m1c_v2_zincir.sh` (`55_m1c_train.py --veri gercek4 --epochs 12 --batch 32`, sonra `--only D2_olcek_rot90 --seeds 1 2`); her tohum için `56_m1c_roma.py --model D2_olcek_rot90[_s1,_s2] --sec`; `57_m1c_2e.py`; `58_m1c_ens.py` | [docs/faz2_m1.md](docs/faz2_m1.md) |
| 3 M2 | `python scripts/60_m2_features.py --splits train calib`; `python scripts/61_m2_score.py` | [docs/faz3_m2.md](docs/faz3_m2.md) |
| 4 M3 | `python scripts/64_m3_conformal.py`; `66_m3_decoy_pairs.py` + önbellek; `65_m3_decoy.py --real`; `67_m3_pipeline.py --export` | [docs/faz4_m3.md](docs/faz4_m3.md) |
| 5 M4 | `python scripts/70_m4_sequences.py`; `71_m4_measure.py`; `72_m4_filter.py --tune --dry --gamma --traj`; `73_m4_figures.py` | [docs/faz5_m4.md](docs/faz5_m4.md) |
| 6 test | `python scripts/80_final_prep.py`; `python scripts/81_freeze.py`; `python scripts/90_ablation.py --final` | [docs/faz6_test.md](docs/faz6_test.md) |
| 6 hız | `python scripts/91_speed.py --n 60` | `results/faz6/speed.md` |

**Test adımı hakkında:** `90_ablation.py --final`, `configs/final_manifest.json`'daki sha256'larla dondurulmuş
girdileri doğrular ve her değerlendirmeden sonra kilit bırakır (`guvenloc/final_guard.py`, F6-1). Kendi makinenizde
yeniden ürettiğiniz ara dosyalar bayt düzeyinde aynı olmayabilir; bu durumda önce `81_freeze.py` ile kendi
manifestinizi yazın. Bizim manifestimiz pakette `params/final_manifest.json` olarak durur. Aynı tablo kodunu
test bölgelerine dokunmadan denemek için: `python scripts/90_ablation.py --rehearse` (calib).

Testler: `python -m pytest -q` (GPU testleri CI'da atlanır).

## Lisans

Kod **MIT** ([LICENSE](LICENSE)). Veri kümesi, RoMa v2 ve DINOv3 bu depoda dağıtılmaz, kendi koşullarıyla edinilir;
ayrıntı ve atıf: [THIRD_PARTY.md](THIRD_PARTY.md). UAV-VisLoc'un lisansı belirtilmediği için depoda ve sürüm paketinde
veri kümesinden hiçbir görüntü yoktur; M1 ağırlıkları ve meta veri yalnız araştırma amaçlıdır (F6-6).
