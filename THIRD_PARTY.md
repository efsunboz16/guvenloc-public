# Üçüncü taraf bileşenler ve veri

`LICENSE` (MIT) yalnızca bu depodaki **kodu** kapsar. Aşağıdakiler bu depoda dağıtılmaz; her biri kendi
koşullarıyla ayrıca edinilir (F6-6, `docs/faz6_test.md`).

| Bileşen | Kullanım | Lisans | Depoda / pakette |
|---|---|---|---|
| [RoMa v2](https://github.com/Parskatt/RoMaV2) (`romav2` 2.0.1) | donuk eşleştirici | MIT | yok; `pip`, ağırlıklar ilk çalıştırmada indirilir |
| [DINOv3](https://github.com/facebookresearch/dinov3) | RoMa v2'nin omurgası | DINOv3 License (Meta) | yok; `torch.hub` ile indirilir, koşulları kullanıcı kabul eder |
| [UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc) (Xu vd., 2024) | drone kareleri, uydu haritaları, GPS | **belirtilmemiş** (GitHub deposunda lisans yok, README yalnız atıf istiyor; arXiv sayfasındaki CC0 makale metni içindir, veriyi kapsamaz); uydu haritaları Google Maps kaynaklı | görüntü yok (tek istisna: README'deki üç tanıtım ekran görüntüsü, atıfla); yalnız dosya adı listeleri, bölge ayrımı ve bizim türettiğimiz geometri/etiketler (sürüm paketi `meta/`) |
| MegaDepth-1500, WxBS | Faz 1 tekrarlama kıyasları | kendi koşulları | yok; `scripts/10_download_data.sh` yönergesi |

**Modül 1 ağırlıkları** (sürüm paketi `weights/`) sıfırdan, UAV-VisLoc eğitim bölgelerinin (01, 08, 09, 10) kareleriyle
eğitilmiş küçük bir CNN'dir; önceden eğitilmiş bir modelden türetilmemiştir. Veri kümesinin lisansı belirtilmediği
için ağırlıklar ve meta veri **yalnız araştırma amaçlı** paylaşılır; kullanımda UAV-VisLoc'a atıf yapılmalıdır.

Veri kümesini kullanırsanız:

```bibtex
@article{xu2024uavvisloc,
  title   = {UAV-VisLoc: A Large-scale Dataset for UAV Visual Localization},
  author  = {Xu, Wenjia and Yao, Yaxuan and Cao, Jiaqi and Wei, Zhiwei and Liu, Chunbo and Wang, Jiuniu and Peng, Mugen},
  journal = {arXiv preprint arXiv:2405.11936},
  year    = {2024}
}
```
