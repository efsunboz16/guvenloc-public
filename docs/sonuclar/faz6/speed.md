# Faz 6B — hız ve bellek (NVIDIA GeForce RTX 3090, RoMa `base`, 60 calib karesi)

| bileşen | ms (medyan) |
|---|---|
| uydu karosu kırpma | 95.0 |
| M1-C (3 ağ × 4 TTA) | 42.5 |
| RoMa v2 eşleştirme | 259.5 |
| RANSAC + özellikler (CPU) | 12.3 |
| oylama hacmi + metrikler (M2) | 53.2 |
| S + conformal (M3) | 0.1 |
| drone↔drone RoMa (M4 odometri) | 221.3 |
| parçacık filtresi adımı (N = 2.000) | 0.4 |

| satır | ms/kare |
|---|---|
| 1 | 367 |
| 2 | 1467 |
| 3 | 1467 |
| 4 | 796 |
| 5 | 462 |
| 6 | 463 |
| 7 | 684 |
| 7-γ | 547 |

En yüksek GPU belleği (PyTorch, RoMa + M1-C + M2 aynı süreçte): **3.59 GB** → 8 GB'lık kartta çalışır.
M1 çağrı tasarrufu: satır 4 2.05 çağrı/kare, satır 2/3: 4 çağrı/kare (RoMa başına 367 ms → kare başına 672 ms kazanç, M1 maliyeti dahil).
γ = 50 m ile uydu yolu kalibrasyonda karelerin %30 oranında atlanır: satır 7 684 → 547 ms/kare.
