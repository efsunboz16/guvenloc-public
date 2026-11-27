# GüvenLoc

Drone görüntülerini uydu haritalarıyla eşleştirerek konum ve yön tahmini yapan
araştırma projesi.

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

Veri, önbellek ve sonuç dosyaları `.gitignore` ile dışarıda tutulur (`results/runs.csv` hariç);
veri lisansları doğrulanmadan paylaşılmaz.

## Lisans

Lisans kararı, veri setlerinin ve kullanılan modellerin lisans koşulları
incelendikten sonra verilecektir.
