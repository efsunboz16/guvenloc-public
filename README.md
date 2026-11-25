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
4. CI yeşil olmadan ve ikinci sorumlu incelemesi tamamlanmadan birleştirmeyin.

## Çalışma ortamı

Bütün GPU işleri (önbellek, tekrarlama, eğitim, değerlendirme) **üniversite sunucusunda** yapılır:
RTX 3090 24 GB, Ubuntu 22.04. Veri, önbellek ve sonuçlar yalnızca orada durur. Kendi bilgisayarınızda
kod yazıp CPU testlerini koşmanız yeterli (`pip install -r requirements.txt` + `pytest -m "not gpu"`).

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
  Faz 1'de çözüldü. Açık tek iş: sunucuya SSH erişimi (MIMARI 0.3).
- **Faz 1** (`faz1/altyapi`, PR #2): veri doğrulama, coğrafi altyapı, bölge ayrımı + çiftler, `matcher.py` +
  önbellek, metrikler/istatistik, taban çizgileri (calib) ve WxBS tekrarlaması tamam.
  Kalan (sunucuda, tek komut `python scripts/faz1_3090.py`): MegaDepth-1500/C4 tekrarlaması ve tam önbellek
  `cache/v1` → `v0.0`. Kılavuz: [docs/faz1_kalan_isler.md](docs/faz1_kalan_isler.md).
  **Özet, sorunlar ve bekleyen kararlar: [docs/faz1_durum.md](docs/faz1_durum.md).**
  Ayrıntı ve komutlar: [docs/faz1_tekrarlama.md](docs/faz1_tekrarlama.md),
  [docs/faz1_kalan_isler.md](docs/faz1_kalan_isler.md).

Veri, önbellek ve sonuç dosyaları `.gitignore` ile dışarıda tutulur (`results/runs.csv` hariç);
veri lisansları doğrulanmadan paylaşılmaz.

## Lisans

Lisans kararı, veri setlerinin ve kullanılan modellerin lisans koşulları
incelendikten sonra verilecektir.
