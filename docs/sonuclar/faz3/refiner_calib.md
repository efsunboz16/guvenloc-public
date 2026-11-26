# Faz 3D — öğrenilmiş inceltici, calib (4a dönmesi, M1-C v0.1)

| tohum | ana skor (val) | val AUROC | satır | AUROC (kendi etiketi) | AUROC (RANSAC etiketi) | başarı tavanı |
|---|---|---|---|---|---|---|
| 0 | s25 | 0.9481 | 4a | 0.9437 | 0.9437 | 0.253 |
| 0 | s25 | 0.9481 | 5a_v0 | 0.9401 | 0.9400 | 0.233 |
| 0 | s25 | 0.9481 | 5r | 0.9513 | 0.9418 | 0.217 |

Tohum 0: AUROC farkı 5r − 4a +0.0075 [-0.0030, +0.0182]; aynı etiketle -0.0020 [-0.0123, +0.0089]. Eşit kabul oranında (%22.9): yanlış kabul 4a 0.144 / 5r 0.155, başarı@25 0.247 / 0.216, McNemar p = 7.75e-02 (yalnız 4a yanlış kabul 148, yalnız 5r 181).
