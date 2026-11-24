# Faz 1 — kalan işler (RTX 3090 makinesi için çalışma kılavuzu)

Bu işler model **eğitmez**; RoMa v2 donuktur. İki ayrı iş var:

| İş | Ne yapar | Bizim kararlarımıza bağlı mı? |
|---|---|---|
| A. RoMa v2 tekrarlaması (MegaDepth-1500, C4) | Makaledeki sayıların bizim kurulumumuzda çıktığını gösterir | **Hayır** — standart benchmark, hemen koşulabilir |
| B. Önbellek `cache/v1` | Bizim drone↔uydu çiftlerimiz için RoMa eşleşmelerini bir kez hesaplar; M1–M4 bunları okur | **Evet** — karonun boyutu/konumu eşleşmeyi değiştirir |

## 0. Önbellekten ÖNCE verilmesi gereken karar

Önbellekteki her dosya "bu drone karesi ↔ şu uydu karosu" eşleşmesidir. Karonun nasıl kesildiği
(`configs/pairs.yaml`) değişirse RoMa'ya giden görüntü değişir ve bütün önbellek yeniden üretilmelidir.

- [ ] `tile_size_m` (şu an 1400 m, GEÇİCİ): RoMa'ya verilen uydu karosunun kenarı. Büyük karo = GNSS
      hatasına daha toleranslı ama drone ile uydu arasındaki ölçek farkı büyür. 30 karelik denemede
      1000 m biraz daha iyiydi (başarı@25 0,20 vs 0,13; anlamlı değil).
- [ ] `tile_offset_m` (şu an ±200 m, GEÇİCİ): karo merkezinin gerçek konumdan en fazla kayması
      ("son bilinen konum + IMU sapması" varsayımı).

Karar değişirse: `configs/pairs.yaml` → `python scripts/14_make_pairs.py data/UAV_VisLoc_dataset`
→ yeni `pairs_v1.csv` (checksum'ı `docs/checksums/pairs_v1.sha256`'ya) → önbellek.
GT kayması kararı (kamera eğimi; `docs/faz1_tekrarlama.md`) önbelleği ETKİLEMEZ, yalnızca değerlendirmeyi.

## 1. Kurulum (Windows veya Linux, Python 3.12)

```
git clone https://github.com/efsunboz16/guvenloc.git && cd guvenloc
git checkout faz1/altyapi            # PR #2 birleşince: main
py -3.12 -m venv .venv               # Linux: python3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1         # Linux: source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python scripts/00_check_env.py       # RTX 3090: sm_86, 24 GB
python -m pytest -q                  # GPU testleri dahil hepsi yeşil olmalı
```

Veri: UAV-VisLoc paketini `data/UAV_VisLoc_dataset/` altına kopyalayın, sonra doğrulayın:
```
python scripts/11_verify_data.py data/UAV_VisLoc_dataset --expect-frames 6774 --expect-maps 11 --checksums
git diff --stat docs/checksums/      # fark çıkmamalı (aynı dosyalar)
python scripts/14_make_pairs.py data/UAV_VisLoc_dataset
sha256sum data/pairs/pairs_v1.csv    # docs/checksums/pairs_v1.sha256 ile aynı olmalı
```

Önce hızı ölçün (süre tahminleri bununla güncellenir):
```
python scripts/23_speed.py data/UAV_VisLoc_dataset --settings base mega1500 precise
```

## 2. A — RoMa v2 tekrarlaması (bağımsız, hemen)

Veri: romav2 deposundaki `scripts/eval_prep.sh`'in yalnızca MegaDepth satırları → `data/megadepth/`
(5 sahne `.npz` + görüntüler). ScanNet indirilmez.
```
python scripts/20_repro_mega1500.py --data data/megadepth --limit 5 --check-original   # duman testi: "birebir aynı: True"
python scripts/20_repro_mega1500.py --data data/megadepth --seeds 0 1 2 3 4            # hedef 62,8 / 77,0 / 86,6 ± 1
python scripts/20_repro_mega1500.py --data data/megadepth --c4 --seeds 0 1 2 3 4       # hedef AUC@5 düşüşü (60,0 → 53,45)
```
Süre: RTX 5050 Laptop'ta ≈ 4 sa/tohum; 3090'da daha kısa (önce 23_speed ile ölçün).
Sonuç: `results/faz1/repro_mega1500.json`, `repro_c4.json` + `results/runs.csv`. Sapma > 1 puan ise
MIMARI 1D'deki B planı (çözünürlük, RANSAC eşiği, sürüm, 5 tohum raporlanır).

## 3. B — Tam önbellek `cache/v1` (karar verildikten sonra)

`configs/base.yaml`: `paths.cache: cache/v1/`, `romav2.setting_cache: mega1500`. Komutlar (sıra önemli:
geliştirme için gereken kümeler önce):
```
python scripts/30_build_cache.py data/UAV_VisLoc_dataset --split calib
python scripts/30_build_cache.py data/UAV_VisLoc_dataset --split train
python scripts/30_build_cache.py data/UAV_VisLoc_dataset --seq            # Modül 4: ardışık kareler
python scripts/30_build_cache.py data/UAV_VisLoc_dataset --split test     # son değerlendirme için
python scripts/31_verify_cache.py cache/v1 --delete-bad                    # her kesintiden sonra
python -m pytest -q tests/test_cache_schema.py
```
- Boyut: 19.778 çift × 4 yön + ≈ 6,7 bin ardışık kare ≈ 86 bin dosya, ≈ 20–22 GB.
- Süre: mega1500, RTX 5050 Laptop'ta ≈ 4,4 s/dosya (≈ 105 sa); 3090'da ölçülen hıza göre hesaplayın.
- **Kesinti güvenli:** betik var olan dosyaları atlar; elektrik kesilir / durdurulursa önce
  `31_verify_cache.py --delete-bad`, sonra aynı komut. Bilgisayarı uyku moduna almayın.
- Dağıtım: `cache/v1/` klasörü (MANIFEST.csv dahil) diğer makinelere kopyalanır; alan kişi
  `31_verify_cache.py cache/v1` ile doğrular.

## 4. Önbellekten sonra (CPU)

```
python scripts/40_baselines.py feats  --cache cache/v1 --split train --ransac-px 3 5 8
python scripts/40_baselines.py feats  --cache cache/v1 --split calib --ransac-px 3 5 8
python scripts/40_baselines.py select --cache cache/v1      # tam eğitim kümesinde eşikler
python scripts/40_baselines.py eval   --cache cache/v1 --split calib
```
Tablo → `docs/faz1_tekrarlama.md` (dev_base tablosunun yerine), `docs/sonuclar/faz1/`.

- [ ] `sigma_scale` kalibrasyonu (Faz 0.4 B planı): gerçek drone↔uydu eşleşmelerinde Σ ölçeği, eğitim
      bölgelerinde NLL ile → `configs/base.yaml: romav2.sigma_scale`. Betiği henüz yazılmadı (Modül 2'den önce).

## 5. Kapanış

- [ ] `docs/faz1_tekrarlama.md` tablolarını doldur, kontrol listesini kapat
- [ ] PR #2 ikinci sorumlu incelemesi → `main`
- [ ] `git tag v0.0 && git push origin v0.0`
- [ ] Danışmana tek sayfalık özet
