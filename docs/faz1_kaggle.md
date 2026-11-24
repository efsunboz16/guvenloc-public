# Faz 1 — kalan işler Kaggle'da (3090 yerine)

> 3090 müsait değil; kalan GPU işleri Kaggle'a taşındı. İşlerin listesi ve sırası değişmedi:
> [faz1_kalan_isler.md](faz1_kalan_isler.md). Bu belge yalnızca **Kaggle'a özgü** adımları verir.

## 0. Kaggle'ın bizi etkileyen kısıtları

| Kısıt | Değer (hesabınızın ayarlarından teyit edin) | Bize etkisi |
|---|---|---|
| GPU | **T4 ×2** (16 GB, sm_75) veya P100 (16 GB, sm_60) | **T4 ×2 seçin.** RoMa v2 bf16 kullanır; T4'te bf16 donanımda yok, öykünmeyle çalışır (yavaş, sayılar biraz farklı olabilir). P100 daha da eski; cu128 PyTorch'u desteklemeyebilir |
| Haftalık GPU kotası | ≈ 30 sa (değişken) | Tam mega1500 önbelleği bu kotaya sığmayabilir (§3) |
| Oturum | en çok 12 sa | Uzun işler oturumlara bölünür; betikler kaldığı yerden devam eder |
| `/kaggle/input` | salt okunur | Bölge 09'un VRT mozaiği otomatik olarak `~/.cache/guvenloc/vrt`'ye yazılır (`GUVENLOC_VRT_DIR` ile değiştirilebilir) |
| `/kaggle/working` | çıktı olarak saklanır, ≈ 20 GB sınır | Sonuçlar ve küçük dosyalar buraya; büyük ara veriler `/kaggle/temp`'e (saklanmaz) |
| İnternet | ayarlardan açılmalı (telefon doğrulaması) | pip, git clone ve RoMa ağırlıkları için gerekli |

İki T4'ü birlikte kullanmak için her GPU'da ayrı süreç koşulur (`CUDA_VISIBLE_DEVICES`).
`30_build_cache.py --shard I/N` ve tohum bazında kaydeden `20_repro_mega1500.py` bunun için yazıldı.

## 1. Veri (bir kez)

- **UAV-VisLoc:** paketi açılmış hâliyle *özel* bir Kaggle Dataset olarak yükleyin, ekip üyelerini
  ortak çalışan olarak ekleyin. Dataset'i not defterine ekleyince yol `/kaggle/input/<ad>/...` olur;
  `find_root` tek alt klasöre kendisi iner.
- **MegaDepth-1500 test verisi:** her oturumda not defterinin içinde indirin (romav2 deposundaki
  `scripts/eval_prep.sh`'in yalnızca MegaDepth satırları) → `/kaggle/temp/megadepth`. Ya da bir kez
  indirip ayrı bir Dataset olarak yükleyin (sonraki oturumlar daha hızlı açılır).

## 2. Her oturumun ilk hücreleri

```bash
%%bash
cd /kaggle/working
git clone -q https://github.com/efsunboz16/guvenloc.git && cd guvenloc && git checkout -q faz1/altyapi
pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -q -r requirements.txt
python scripts/00_check_env.py          # iki T4 görünmeli; sm_75 için bf16 UYARI'sı beklenir
python -m pytest -q                     # GPU testleri dahil hepsi yeşil olmalı
```
cu128 sürümü Kaggle'ın sürücüsünde açılmazsa: aynı komutta `cu128` → `cu126`. Hangi sürümü
kullandığınızı not edin (`results/runs.csv` zaten torch sürümünü kaydeder).

Önceki oturumun çıktısını ("Add Input → Notebook output") eklediyseniz sonuçları geri taşıyın:
```bash
%%bash
cd /kaggle/working/guvenloc
cp -n /kaggle/input/<onceki-cikti>/guvenloc/results/faz1/*.json results/faz1/ 2>/dev/null || true
cp /kaggle/input/<onceki-cikti>/guvenloc/results/runs.csv results/runs.csv 2>/dev/null || true
```

**İlk oturumda bir kez: hız ölçümü.** Bütün süre planı buna dayanır.
```bash
%%bash
cd /kaggle/working/guvenloc
UAV=$(ls -d /kaggle/input/*/ | head -1)       # UAV-VisLoc dataset'inin yolu (gerekirse elle yazın)
python scripts/23_speed.py "$UAV" --settings base mega1500
python scripts/20_repro_mega1500.py --data /kaggle/temp/megadepth --limit 5 --check-original
```
- `--check-original` **"birebir aynı: True"** yazmalı. Bu, T4'te bizim döngümüzün romav2'ninkiyle
  aynı olduğunu gösterir. Mutlak sayının 62,8'e yakınlığını ise tam koşu gösterecek.
- `23_speed` çıktısındaki `mega1500 matcher_s` = önbellekte dosya başına süre (t). Bunu §3'teki
  hesapta kullanın.

## 3. Ne nerede koşacak — karar kuralı

Dizüstü (RTX 5050) **hâlâ kullanılabilir** ve Kaggle ile aynı anda çalışabilir. Ölçülen hızlar:
dizüstü mega1500 4,4 s/dosya, base 0,75 s/dosya.

| İş | GPU-saat (dizüstü ölçümü) | Öneri |
|---|---|---|
| A. MegaDepth-1500, 5 tohum | ≈ 5 × 4 sa | **Kaggle** (GPU0) |
| A. MegaDepth-C4, 5 tohum | ≈ 5 × 4 sa | **Kaggle** (GPU1) |
| K2 deneyi (300 eğitim karesi × 2 karo seçeneği × 4 yön) | mega1500 ≈ 3 sa, base ≈ 0,5 sa | önbellekle aynı ayarda |
| B. Tam önbellek (≈ 86 bin dosya) | mega1500 ≈ 105 sa, base ≈ 17 sa | aşağıdaki kurala göre |

**Önbellek kuralı.** T4 ×2'de tam mega1500 önbelleğinin süresi ≈ 86.000 × t / 2 / 3600 saattir.
- **≤ 30 sa** (t ≲ 2,5 s): mega1500 önbelleği Kaggle'da, 2–3 oturumda
  (`--shard 0/2` ve `--shard 1/2`, aşağıda).
- **Daha uzunsa** (beklenen durum; T4 dizüstünden yavaş olabilir): önbellek **dizüstünde `base`
  ayarıyla**, bir gecede (≈ 17 sa). Bu, MIMARI'deki `setting_cache: mega1500` kararını değiştirir.
  Ekip onayı gerekir. Gerekçe: aynı ayar bütün kümelerde kullanılır (eşikler eğitimden calib/test'e
  taşınır); karışık ayar olmaz. mega1500 yalnızca tekrarlamada (A) kalır. Raporda "önbellek
  ayarı: base (640 px), donanım kısıtı" diye belirtilir.

Bütçe örneği (≈ 30 sa/hafta): A için T4'te bir tohum ≈ T saat sürüyorsa iki GPU'da 10 koşu
≈ 5T saat duvar süresi tutar. Haftayı aşarsa tohum sayısını 3'e indirin; ortalama ± std yine raporlanır.

## 4. A — MegaDepth-1500 + C4 (iki GPU paralel)

"Save Version → Save & Run All" ile arka planda koşturun (tarayıcıyı kapatabilirsiniz, 12 sa sınırı).
Her oturumda, `--limit 5` ölçümüne göre 11 saate sığacak sayıda tohum verin:
```bash
%%bash
cd /kaggle/working/guvenloc && mkdir -p logs
MD=/kaggle/temp/megadepth
CUDA_VISIBLE_DEVICES=0 python scripts/20_repro_mega1500.py --data $MD --seeds 0 1 2 3 4 > logs/mega_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python scripts/20_repro_mega1500.py --data $MD --c4 --seeds 0 1 2 3 4 > logs/c4_gpu1.log 2>&1 &
wait; tail -5 logs/*.log
```
- Biten her tohum hemen `results/faz1/repro_<ad>_seed<S>.json`'a yazılır. Oturum 12. saatte kesilse
  bile biten tohumlar kaybolmaz. Sonraki oturumda aynı komut biten tohumları atlar (§2'deki geri
  taşıma hücresiyle).
- Özet (`repro_mega1500.json`, `repro_c4.json`) o ana kadar biten **bütün** tohumlardan üretilir.
- Tamamlanınca `results/faz1/repro_*.json` ve `results/runs.csv`'yi çıktıdan indirip dizüstündeki
  repoya koyun ve commit'leyin. Kaggle'daki klon aynı commit'ten başladığı için `runs.csv`'nin üzerine
  yazmak güvenlidir (arada dizüstünde başka koşu yapıldıysa yalnızca yeni satırları ekleyin).
  Not: GitHub'a doğrudan Kaggle'dan push için Kaggle Secrets'a kişisel token koymak gerekir;
  ekipçe karar verin.

Sapma > 1 puan ise önce T4/bf16 öykünmesinden şüphelenin: `--limit 5` alt kümesini dizüstünde de
koşup iki kartı karşılaştırın (aynı tohum). Fark oradan geliyorsa raporda belirtilir. Gelmiyorsa
MIMARI 1D'deki B planına geçilir.

## 5. B — önbellek Kaggle'da koşacaksa (§3 kuralı izin verirse)

```bash
%%bash
cd /kaggle/working/guvenloc && mkdir -p logs
UAV=/kaggle/input/<uav-visloc>
python scripts/14_make_pairs.py $UAV && sha256sum data/pairs/pairs_v1.csv   # docs/checksums ile aynı olmalı
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$i python scripts/30_build_cache.py $UAV --split calib --shard $i/2 \
      --cache /kaggle/working/cache/v1 > logs/cache_$i.log 2>&1 &
done
wait; python scripts/31_verify_cache.py /kaggle/working/cache/v1 --delete-bad
```
- Her parça kendi `MANIFEST.partial.I.csv`'sine yazar; `--finalize` ve `31_verify_cache.py` hepsini
  birleştirir.
- Çıktı sınırı (≈ 20 GB) yüzünden **her küme ayrı bir oturumda** üretilir (calib ≈ 2 GB, train,
  seq, test). Her biri ayrı bir Kaggle Dataset'e kaydedilir. Yerelde klasörler `cache/v1/` altında
  birleştirilir, ardından `31_verify_cache.py cache/v1` koşulur.

## 6. Kaggle'a bağlı olmayanlar (dizüstünde, aynı anda)

- K2 uyarlamalı karo kodu + deney betiği (yazılacak). Önbellek `base` olursa deney de `base` ile
  dizüstünde koşar (≈ 0,5 sa).
- Önbellek `base` olursa: `python scripts/30_build_cache.py data/UAV_VisLoc_dataset --setting base`
  (calib → train → `--seq` → test). Bilgisayarı uyku moduna almayın; kesilirse
  `31_verify_cache.py --delete-bad` + aynı komut.
- Taban çizgileri, K1 eğim analizi, `sigma_scale` kalibrasyonu: CPU / önbellekten okur.
