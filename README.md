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

## Kurulum (Windows veya Linux, Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux: source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/00_check_env.py
python scripts/01_roma_single_pair.py --setting mega1500
```

RTX 50 serisi (Blackwell, `sm_120`) için cu128 PyTorch şarttır. İlk çalıştırma RoMa v2
ağırlıklarını (~1 GB) ve DINOv3 kodunu `~/.cache/torch/hub` altına indirir.
Modüller arası veri biçimleri [INTERFACES.md](INTERFACES.md) dosyasındadır.

### Gün 1 ölçümleri (RTX 5060 Laptop 8 GB, Windows 11, sentetik çift)

| Ayar | çözünürlük (lr/hr) | süre/çift | en yüksek bellek | köşe hatası |
|---|---|---|---|---|
| precise | 800 / 1280 | 3,7 s | 10,4 GB (8 GB'ı aşar) | 0,31 px |
| mega1500 / wxbs | 800 / 1024 | 2,1 s | 7,1 GB | 0,32 px |
| base | 640 | 0,31 s | 3,4 GB | 0,34 px |
| fast | 512 | 0,20 s | 2,6 GB | 0,33 px |

## Durum

İlk depo iskeleti oluşturulmuştur. Veri, önbellek ve sonuç dosyaları
`.gitignore` ile dışarıda tutulur; veri lisansları doğrulanmadan paylaşılmaz.

## Lisans

Lisans kararı, veri setlerinin ve kullanılan modellerin lisans koşulları
incelendikten sonra verilecektir.
