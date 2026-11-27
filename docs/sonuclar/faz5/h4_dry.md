# Faz 5 — H4 kuru denemesi (calib 05 + 11; test değil), `configs/h4.yaml`

Filtre ayarı eğitimde seçildi (`tune.json`): {'n_particles': 2000, 'sigma_pos0_m': 5.0, 'sigma_pos_rate': 0.1, 'sigma_theta_deg': 1.0, 'jump_p': 0.02, 'jump_sigma_m': 100.0, 'lik_power': 2.0, 'eps': 0.001, 'theta_meas_sigma_deg': 5.0, 'theta_outlier_p': 0.3, 'theta_init_sigma_deg': 5.0, 'resample_frac': 0.5, 'reset_lik': 0.0001, 'seed': 0}.

Satır 6 = tek kare hacim tepesi (her karede), satır 7 = filtre. Azalma = 1 − med7/med6 [blok bootstrap %95]; Wilcoxon tek yönlü. Kapsama: filtre başlamış karelerde hata ≤ κ · %90 yayılım. 'havuz' κ'yı aynı veride seçer (tanım gereği ≈ 0,90); '05 → 11' κ'yı 05'te seçip 11'de ölçer.

| satır | değerlendirme | n | med6 | med7 | azalma [GA] | Wilcoxon p | başarı@25 6/7 | başarı@50 6/7 | κ | kapsama 7 | yarıçap medyan | H4 (i) (ii) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7 | pooled_in_sample | 1063 | 61.5 | 38.3 | 0.376 [0.198, 0.823] | 2.9e-19 | 0.235/0.298 | 0.462/0.608 | 3.25 | 0.900 | 173 m | ✓ ✓ |
| 7 | kappa 05 → 11 | 590 | 53.7 | 34.6 | 0.355 [0.215, 0.639] | 5.5e-19 | 0.249/0.336 | 0.488/0.675 | 3.75 | 0.947 | 176 m | ✓ ✓ |
| 7 | kappa 11 → 05 | 473 | 121.0 | 44.6 | 0.632 [0.081, 0.848] | 5.5e-04 | 0.218/0.252 | 0.429/0.524 | 2.66 | 0.789 | 285 m | ✓ ✗ |
| 7-noupd | pooled_in_sample | 1063 | 61.5 | 40.9 | 0.334 [0.147, 0.801] | 4.4e-20 | 0.235/0.278 | 0.462/0.570 | 3.38 | 0.900 | 181 m | ✓ ✓ |
| 7-noupd | kappa 05 → 11 | 590 | 53.7 | 38.4 | 0.284 [0.127, 0.604] | 2.7e-19 | 0.249/0.300 | 0.488/0.619 | 3.82 | 0.946 | 182 m | ✓ ✓ |
| 7-noupd | kappa 11 → 05 | 473 | 121.0 | 47.4 | 0.608 [0.076, 0.820] | 2.5e-04 | 0.218/0.249 | 0.429/0.510 | 2.65 | 0.746 | 266 m | ✓ ✗ |
| 7-cv | pooled_in_sample | 1063 | 61.5 | 57.4 | 0.067 [-0.123, 0.653] | 9.4e-01 | 0.235/0.231 | 0.462/0.464 | 5.43 | 0.900 | 717 m | ✗ ✓ |
| 7-cv | kappa 05 → 11 | 590 | 53.7 | 48.6 | 0.094 [-0.111, 0.419] | 4.0e-03 | 0.249/0.256 | 0.488/0.508 | 6.57 | 0.936 | 684 m | ✗ ✓ |
| 7-cv | kappa 11 → 05 | 473 | 121.0 | 90.1 | 0.255 [-0.424, 0.659] | 1.0e+00 | 0.218/0.201 | 0.429/0.408 | 4.46 | 0.835 | 808 m | ✗ ✗ |

Bölge başına (satır 7): 05: 121 → 45 m; 11: 54 → 35 m
