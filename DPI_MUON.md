nini@nini-B550M-K:~/DPI$ python3 scripts/duel_dpi_muon_dso_100m.py
⚔️ ULTIMATE DUEL 100M (2000 steps, Cosine LR 1e-e3): DPI+SpectreMuon vs DPI (Multiple versions)+Muon vs Xavier+Muon
================================================================================

🚀 Running DPI v17.0 + SpectreMuon...
  [DPI] mode=V17.0 | scale=1.0000 | anchors=True
  [Phase 0] Seeding Lexical Manifold...
  [Phase 1] Semantic Clustering...
  [Phase 2] Genomic Bootstrapping...
  [Phase 4] Calibrating Output Head...
DPI COMPLETE (Standard).
  [DPI v17.0 + SpectreMuon] Step  100 | Loss: 7.3037 | ERank W_q: 4/768
  [DPI v17.0 + SpectreMuon] Step  200 | Loss: 6.6905 | ERank W_q: 5/768
  [DPI v17.0 + SpectreMuon] Step  300 | Loss: 6.5095 | ERank W_q: 6/768
  [DPI v17.0 + SpectreMuon] Step  400 | Loss: 6.1972 | ERank W_q: 7/768
  [DPI v17.0 + SpectreMuon] Step  500 | Loss: 6.0395 | ERank W_q: 8/768
  [DPI v17.0 + SpectreMuon] Step  600 | Loss: 5.9597 | ERank W_q: 9/768
  [DPI v17.0 + SpectreMuon] Step  700 | Loss: 5.8286 | ERank W_q: 12/768
  [DPI v17.0 + SpectreMuon] Step  800 | Loss: 5.7757 | ERank W_q: 15/768
  [DPI v17.0 + SpectreMuon] Step  900 | Loss: 5.7547 | ERank W_q: 18/768
  [DPI v17.0 + SpectreMuon] Step 1000 | Loss: 5.5854 | ERank W_q: 20/768
  [DPI v17.0 + SpectreMuon] Step 1100 | Loss: 5.6756 | ERank W_q: 21/768
  [DPI v17.0 + SpectreMuon] Step 1200 | Loss: 5.5231 | ERank W_q: 24/768
  [DPI v17.0 + SpectreMuon] Step 1300 | Loss: 5.4253 | ERank W_q: 25/768
  [DPI v17.0 + SpectreMuon] Step 1400 | Loss: 5.4491 | ERank W_q: 27/768
  [DPI v17.0 + SpectreMuon] Step 1500 | Loss: 5.4666 | ERank W_q: 27/768
  [DPI v17.0 + SpectreMuon] Step 1600 | Loss: 5.4954 | ERank W_q: 28/768
  [DPI v17.0 + SpectreMuon] Step 1700 | Loss: 5.4304 | ERank W_q: 28/768
  [DPI v17.0 + SpectreMuon] Step 1800 | Loss: 5.3542 | ERank W_q: 28/768
  [DPI v17.0 + SpectreMuon] Step 1900 | Loss: 5.3232 | ERank W_q: 30/768
  [DPI v17.0 + SpectreMuon] Step 2000 | Loss: 5.3204 | ERank W_q: 30/768

🚀 Running DPI v14 + Muon...
  [Phase 0] Seeding Lexical Manifold (Exact SVD)...
  [Phase 2] Sequential Bootstrapping v15.2 (Attention Arch: False)...
    Layer  0 | QK-Alignment: 0.600
    Layer  5 | QK-Alignment: 0.327
    Layer 10 | QK-Alignment: 0.055
    Layer 11 | QK-Alignment: 0.000
  [Phase 3] Final Manifold Calibration...
DPI-15.2 Attention Arch Initialization Complete.
  [DPI v14 + Muon] Step  100 | Loss: 7.0553 | ERank W_q: 768/768
  [DPI v14 + Muon] Step  200 | Loss: 6.7063 | ERank W_q: 768/768
  [DPI v14 + Muon] Step  300 | Loss: 6.5733 | ERank W_q: 768/768
  [DPI v14 + Muon] Step  400 | Loss: 6.2964 | ERank W_q: 768/768
  [DPI v14 + Muon] Step  500 | Loss: 6.2023 | ERank W_q: 768/768
  [DPI v14 + Muon] Step  600 | Loss: 6.0289 | ERank W_q: 768/768
  [DPI v14 + Muon] Step  700 | Loss: 5.8572 | ERank W_q: 768/768
  [DPI v14 + Muon] Step  800 | Loss: 5.9981 | ERank W_q: 768/768
  [DPI v14 + Muon] Step  900 | Loss: 5.8638 | ERank W_q: 768/768
  [DPI v14 + Muon] Step 1000 | Loss: 5.7447 | ERank W_q: 768/768
  [DPI v14 + Muon] Step 1100 | Loss: 5.6911 | ERank W_q: 768/768
  [DPI v14 + Muon] Step 1200 | Loss: 5.7758 | ERank W_q: 768/768
  [DPI v14 + Muon] Step 1300 | Loss: 5.6468 | ERank W_q: 768/768
  [DPI v14 + Muon] Step 1400 | Loss: 5.6958 | ERank W_q: 768/768
  [DPI v14 + Muon] Step 1500 | Loss: 5.7883 | ERank W_q: 768/768
  [DPI v14 + Muon] Step 1600 | Loss: 5.4821 | ERank W_q: 768/768
  [DPI v14 + Muon] Step 1700 | Loss: 5.4698 | ERank W_q: 768/768
  [DPI v14 + Muon] Step 1800 | Loss: 5.7150 | ERank W_q: 768/768
  [DPI v14 + Muon] Step 1900 | Loss: 5.5714 | ERank W_q: 768/768
  [DPI v14 + Muon] Step 2000 | Loss: 5.5933 | ERank W_q: 768/768

🚀 Running DPI v15 + Muon...
  [Phase 0] Seeding Lexical Manifold (Exact SVD)...
  [Phase 2] Sequential Bootstrapping v15.2 (Attention Arch: True)...
    Layer  0 | QK-Alignment: 0.000
    Layer  5 | QK-Alignment: 0.396
    Layer 10 | QK-Alignment: 0.113
    Layer 11 | QK-Alignment: 0.000
  [Phase 3] Final Manifold Calibration...
DPI-15.2 Attention Arch Initialization Complete.
  [DPI v15 + Muon] Step  100 | Loss: 7.2509 | ERank W_q: 768/768
  [DPI v15 + Muon] Step  200 | Loss: 6.9277 | ERank W_q: 768/768
  [DPI v15 + Muon] Step  300 | Loss: 6.7477 | ERank W_q: 768/768
  [DPI v15 + Muon] Step  400 | Loss: 6.4507 | ERank W_q: 768/768
  [DPI v15 + Muon] Step  500 | Loss: 6.3397 | ERank W_q: 768/768
  [DPI v15 + Muon] Step  600 | Loss: 6.1690 | ERank W_q: 768/768
  [DPI v15 + Muon] Step  700 | Loss: 5.9730 | ERank W_q: 768/768
  [DPI v15 + Muon] Step  800 | Loss: 6.1150 | ERank W_q: 768/768
  [DPI v15 + Muon] Step  900 | Loss: 5.9996 | ERank W_q: 768/768
  [DPI v15 + Muon] Step 1000 | Loss: 5.9086 | ERank W_q: 768/768
  [DPI v15 + Muon] Step 1100 | Loss: 5.8479 | ERank W_q: 768/768
  [DPI v15 + Muon] Step 1200 | Loss: 5.9309 | ERank W_q: 768/768
  [DPI v15 + Muon] Step 1300 | Loss: 5.8063 | ERank W_q: 768/768
  [DPI v15 + Muon] Step 1400 | Loss: 5.8623 | ERank W_q: 768/768
  [DPI v15 + Muon] Step 1500 | Loss: 5.9414 | ERank W_q: 768/768
  [DPI v15 + Muon] Step 1600 | Loss: 5.6646 | ERank W_q: 768/768
  [DPI v15 + Muon] Step 1700 | Loss: 5.6398 | ERank W_q: 768/768
  [DPI v15 + Muon] Step 1800 | Loss: 5.8762 | ERank W_q: 768/768
  [DPI v15 + Muon] Step 1900 | Loss: 5.7461 | ERank W_q: 768/768
  [DPI v15 + Muon] Step 2000 | Loss: 5.7547 | ERank W_q: 768/768

🚀 Running DPI v16.2 + Muon...
  [DPI] mode=V16.2 | scale=1.0000 | anchors=False
  [Phase 0] Seeding Lexical Manifold...
  [Phase 1] Semantic Clustering...
  [Phase 2] Genomic Bootstrapping...
  [Phase 4] Calibrating Output Head...
DPI COMPLETE (Standard).
  [DPI v16.2 + Muon] Step  100 | Loss: 7.2041 | ERank W_q: 6/768
  [DPI v16.2 + Muon] Step  200 | Loss: 6.4962 | ERank W_q: 19/768
  [DPI v16.2 + Muon] Step  300 | Loss: 6.3290 | ERank W_q: 41/768
  [DPI v16.2 + Muon] Step  400 | Loss: 5.9838 | ERank W_q: 70/768
  [DPI v16.2 + Muon] Step  500 | Loss: 5.8194 | ERank W_q: 142/768
  [DPI v16.2 + Muon] Step  600 | Loss: 5.7230 | ERank W_q: 261/768
  [DPI v16.2 + Muon] Step  700 | Loss: 5.5692 | ERank W_q: 368/768
  [DPI v16.2 + Muon] Step  800 | Loss: 5.5673 | ERank W_q: 437/768
  [DPI v16.2 + Muon] Step  900 | Loss: 5.5101 | ERank W_q: 480/768
  [DPI v16.2 + Muon] Step 1000 | Loss: 5.3589 | ERank W_q: 504/768
  [DPI v16.2 + Muon] Step 1100 | Loss: 5.4625 | ERank W_q: 522/768
  [DPI v16.2 + Muon] Step 1200 | Loss: 5.2934 | ERank W_q: 532/768
  [DPI v16.2 + Muon] Step 1300 | Loss: 5.2062 | ERank W_q: 540/768
  [DPI v16.2 + Muon] Step 1400 | Loss: 5.2289 | ERank W_q: 545/768
  [DPI v16.2 + Muon] Step 1500 | Loss: 5.2758 | ERank W_q: 549/768
  [DPI v16.2 + Muon] Step 1600 | Loss: 5.2735 | ERank W_q: 551/768
  [DPI v16.2 + Muon] Step 1700 | Loss: 5.2336 | ERank W_q: 553/768
  [DPI v16.2 + Muon] Step 1800 | Loss: 5.1474 | ERank W_q: 554/768
  [DPI v16.2 + Muon] Step 1900 | Loss: 5.1013 | ERank W_q: 554/768
  [DPI v16.2 + Muon] Step 2000 | Loss: 5.1176 | ERank W_q: 554/768

🚀 Running DPI v16.3 + Muon...
  [DPI] mode=V16.3 | scale=1.0000 | anchors=False
  [Phase 0] Seeding Lexical Manifold...
  [Phase 1] Semantic Clustering...
  [Phase 2] Genomic Bootstrapping...
  [Phase 4] Calibrating Output Head...
DPI COMPLETE (Standard).
  [DPI v16.3 + Muon] Step  100 | Loss: 7.2041 | ERank W_q: 6/768
  [DPI v16.3 + Muon] Step  200 | Loss: 6.4962 | ERank W_q: 19/768
  [DPI v16.3 + Muon] Step  300 | Loss: 6.3290 | ERank W_q: 41/768
  [DPI v16.3 + Muon] Step  400 | Loss: 5.9838 | ERank W_q: 70/768
  [DPI v16.3 + Muon] Step  500 | Loss: 5.8194 | ERank W_q: 142/768
  [DPI v16.3 + Muon] Step  600 | Loss: 5.7230 | ERank W_q: 261/768
  [DPI v16.3 + Muon] Step  700 | Loss: 5.5692 | ERank W_q: 368/768
  [DPI v16.3 + Muon] Step  800 | Loss: 5.5673 | ERank W_q: 437/768
  [DPI v16.3 + Muon] Step  900 | Loss: 5.5101 | ERank W_q: 480/768
  [DPI v16.3 + Muon] Step 1000 | Loss: 5.3589 | ERank W_q: 504/768
  [DPI v16.3 + Muon] Step 1100 | Loss: 5.4625 | ERank W_q: 522/768
  [DPI v16.3 + Muon] Step 1200 | Loss: 5.2934 | ERank W_q: 532/768
  [DPI v16.3 + Muon] Step 1300 | Loss: 5.2062 | ERank W_q: 540/768
  [DPI v16.3 + Muon] Step 1400 | Loss: 5.2289 | ERank W_q: 545/768
  [DPI v16.3 + Muon] Step 1500 | Loss: 5.2758 | ERank W_q: 549/768
  [DPI v16.3 + Muon] Step 1600 | Loss: 5.2735 | ERank W_q: 551/768
  [DPI v16.3 + Muon] Step 1700 | Loss: 5.2336 | ERank W_q: 553/768
  [DPI v16.3 + Muon] Step 1800 | Loss: 5.1474 | ERank W_q: 554/768
  [DPI v16.3 + Muon] Step 1900 | Loss: 5.1013 | ERank W_q: 554/768
  [DPI v16.3 + Muon] Step 2000 | Loss: 5.1176 | ERank W_q: 554/768

🚀 Running DPI v17.0 + Muon...
  [DPI] mode=V17.0 | scale=1.0000 | anchors=True
  [Phase 0] Seeding Lexical Manifold...
  [Phase 1] Semantic Clustering...
  [Phase 2] Genomic Bootstrapping...
  [Phase 4] Calibrating Output Head...
DPI COMPLETE (Standard).
  [DPI v17.0 + Muon] Step  100 | Loss: 7.1901 | ERank W_q: 7/768
  [DPI v17.0 + Muon] Step  200 | Loss: 6.5187 | ERank W_q: 23/768
  [DPI v17.0 + Muon] Step  300 | Loss: 6.3243 | ERank W_q: 42/768
  [DPI v17.0 + Muon] Step  400 | Loss: 5.9872 | ERank W_q: 75/768
  [DPI v17.0 + Muon] Step  500 | Loss: 5.8205 | ERank W_q: 147/768
  [DPI v17.0 + Muon] Step  600 | Loss: 5.7279 | ERank W_q: 266/768
  [DPI v17.0 + Muon] Step  700 | Loss: 5.5668 | ERank W_q: 372/768
  [DPI v17.0 + Muon] Step  800 | Loss: 5.5607 | ERank W_q: 437/768
  [DPI v17.0 + Muon] Step  900 | Loss: 5.5235 | ERank W_q: 479/768
  [DPI v17.0 + Muon] Step 1000 | Loss: 5.3667 | ERank W_q: 505/768
  [DPI v17.0 + Muon] Step 1100 | Loss: 5.4639 | ERank W_q: 521/768
  [DPI v17.0 + Muon] Step 1200 | Loss: 5.2984 | ERank W_q: 533/768
  [DPI v17.0 + Muon] Step 1300 | Loss: 5.1995 | ERank W_q: 540/768
  [DPI v17.0 + Muon] Step 1400 | Loss: 5.2311 | ERank W_q: 545/768
  [DPI v17.0 + Muon] Step 1500 | Loss: 5.2656 | ERank W_q: 549/768
  [DPI v17.0 + Muon] Step 1600 | Loss: 5.2787 | ERank W_q: 552/768
  [DPI v17.0 + Muon] Step 1700 | Loss: 5.2381 | ERank W_q: 553/768
  [DPI v17.0 + Muon] Step 1800 | Loss: 5.1467 | ERank W_q: 555/768
  [DPI v17.0 + Muon] Step 1900 | Loss: 5.1125 | ERank W_q: 555/768
  [DPI v17.0 + Muon] Step 2000 | Loss: 5.1322 | ERank W_q: 555/768

🚀 Running Xavier + Muon...
  [Xavier+Muon] Step  100 | Loss: 7.5659 | ERank W_q: 758/768
  [Xavier+Muon] Step  200 | Loss: 6.5677 | ERank W_q: 758/768
  [Xavier+Muon] Step  300 | Loss: 6.1671 | ERank W_q: 759/768
  [Xavier+Muon] Step  400 | Loss: 6.0997 | ERank W_q: 759/768
  [Xavier+Muon] Step  500 | Loss: 6.0439 | ERank W_q: 758/768
  [Xavier+Muon] Step  600 | Loss: 5.9859 | ERank W_q: 758/768
  [Xavier+Muon] Step  700 | Loss: 6.0797 | ERank W_q: 758/768
  [Xavier+Muon] Step  800 | Loss: 5.9319 | ERank W_q: 758/768
  [Xavier+Muon] Step  900 | Loss: 5.7915 | ERank W_q: 758/768
  [Xavier+Muon] Step 1000 | Loss: 5.8109 | ERank W_q: 757/768
  [Xavier+Muon] Step 1100 | Loss: 5.7205 | ERank W_q: 757/768
  [Xavier+Muon] Step 1200 | Loss: 5.7274 | ERank W_q: 758/768
  [Xavier+Muon] Step 1300 | Loss: 5.8217 | ERank W_q: 758/768
  [Xavier+Muon] Step 1400 | Loss: 5.7523 | ERank W_q: 758/768
  [Xavier+Muon] Step 1500 | Loss: 5.7173 | ERank W_q: 758/768
  [Xavier+Muon] Step 1600 | Loss: 5.7171 | ERank W_q: 758/768
  [Xavier+Muon] Step 1700 | Loss: 5.7902 | ERank W_q: 758/768
  [Xavier+Muon] Step 1800 | Loss: 5.6025 | ERank W_q: 758/768
  [Xavier+Muon] Step 1900 | Loss: 5.7839 | ERank W_q: 758/768
  [Xavier+Muon] Step 2000 | Loss: 5.6176 | ERank W_q: 757/768

============================================================
Configuration                  | Val Loss
------------------------------------------------------------
DPI_v17.0_SpectreMuon          | 5.2773
DPI_v14_Muon                   | 5.4188
DPI_v15_Muon                   | 5.6016
DPI_v16.2_Muon                 | 5.0636
DPI_v16.3_Muon                 | 5.0636
DPI_v17.0_Muon                 | 5.0639
Xavier_Muon                    | 5.6572
============================================================







#20M 20k
⚔️ ULTIMATE DUEL 20M (20k steps, Cosine LR 1e-3)
================================================================================

🚀 Running Xavier + Muon...
  [Xavier+Muon] Step   500 | Loss: 8.6425 | ERank W_q: 314/320
  [Xavier+Muon] Step  1000 | Loss: 6.3914 | ERank W_q: 307/320
  [Xavier+Muon] Step  1500 | Loss: 6.1083 | ERank W_q: 301/320
  [Xavier+Muon] Step  2000 | Loss: 5.9467 | ERank W_q: 294/320
  [Xavier+Muon] Step  2500 | Loss: 5.8359 | ERank W_q: 288/320
  [Xavier+Muon] Step  3000 | Loss: 5.7081 | ERank W_q: 284/320
  [Xavier+Muon] Step  3500 | Loss: 5.5446 | ERank W_q: 281/320
  [Xavier+Muon] Step  4000 | Loss: 5.5544 | ERank W_q: 277/320
  [Xavier+Muon] Step  4500 | Loss: 5.5017 | ERank W_q: 278/320
  [Xavier+Muon] Step  5000 | Loss: 5.4329 | ERank W_q: 276/320
  [Xavier+Muon] Step  5500 | Loss: 5.2387 | ERank W_q: 275/320
  [Xavier+Muon] Step  6000 | Loss: 5.3239 | ERank W_q: 275/320
  [Xavier+Muon] Step  6500 | Loss: 5.3519 | ERank W_q: 275/320
  [Xavier+Muon] Step  7000 | Loss: 5.3819 | ERank W_q: 275/320
  [Xavier+Muon] Step  7500 | Loss: 5.2529 | ERank W_q: 277/320
  [Xavier+Muon] Step  8000 | Loss: 5.1525 | ERank W_q: 278/320
  [Xavier+Muon] Step  8500 | Loss: 5.2154 | ERank W_q: 278/320
  [Xavier+Muon] Step  9000 | Loss: 5.2316 | ERank W_q: 281/320
  [Xavier+Muon] Step  9500 | Loss: 5.1643 | ERank W_q: 281/320
  [Xavier+Muon] Step 10000 | Loss: 5.0970 | ERank W_q: 281/320
  [Xavier+Muon] Step 10500 | Loss: 5.4571 | ERank W_q: 283/320
  [Xavier+Muon] Step 11000 | Loss: 5.2400 | ERank W_q: 285/320
  [Xavier+Muon] Step 11500 | Loss: 5.2160 | ERank W_q: 285/320
  [Xavier+Muon] Step 12000 | Loss: 4.9215 | ERank W_q: 286/320
  [Xavier+Muon] Step 12500 | Loss: 4.8011 | ERank W_q: 286/320
  [Xavier+Muon] Step 13000 | Loss: 5.2119 | ERank W_q: 288/320
  [Xavier+Muon] Step 13500 | Loss: 5.0288 | ERank W_q: 290/320
  [Xavier+Muon] Step 14000 | Loss: 5.1814 | ERank W_q: 290/320
  [Xavier+Muon] Step 14500 | Loss: 4.7266 | ERank W_q: 290/320
  [Xavier+Muon] Step 15000 | Loss: 5.0848 | ERank W_q: 291/320
  [Xavier+Muon] Step 15500 | Loss: 4.8464 | ERank W_q: 291/320
  [Xavier+Muon] Step 16000 | Loss: 5.1426 | ERank W_q: 290/320
  [Xavier+Muon] Step 16500 | Loss: 5.0381 | ERank W_q: 290/320
  [Xavier+Muon] Step 17000 | Loss: 5.0269 | ERank W_q: 290/320
  [Xavier+Muon] Step 17500 | Loss: 4.8973 | ERank W_q: 291/320
  [Xavier+Muon] Step 18000 | Loss: 4.9441 | ERank W_q: 291/320
  [Xavier+Muon] Step 18500 | Loss: 5.0972 | ERank W_q: 292/320
  [Xavier+Muon] Step 19000 | Loss: 4.8517 | ERank W_q: 292/320
  [Xavier+Muon] Step 19500 | Loss: 5.0307 | ERank W_q: 291/320
  [Xavier+Muon] Step 20000 | Loss: 4.9607 | ERank W_q: 292/320

🚀 Running DPI v17.0 + SpectreMuon...
  [DPI] mode=V17.0 | scale=1.0000 | anchors=True
  [Phase 0] Seeding Lexical Manifold...
  [Phase 1] Semantic Clustering...
  [Phase 2] Genomic Bootstrapping...
  [Phase 4] Calibrating Output Head...
DPI COMPLETE (Standard).
  [DPI v17.0 + SpectreMuon] Step   500 | Loss: 6.9389 | ERank W_q: 9/320
  [DPI v17.0 + SpectreMuon] Step  1000 | Loss: 6.1643 | ERank W_q: 23/320
  [DPI v17.0 + SpectreMuon] Step  1500 | Loss: 5.5267 | ERank W_q: 72/320
  [DPI v17.0 + SpectreMuon] Step  2000 | Loss: 5.5210 | ERank W_q: 107/320
  [DPI v17.0 + SpectreMuon] Step  2500 | Loss: 5.5018 | ERank W_q: 131/320
  [DPI v17.0 + SpectreMuon] Step  3000 | Loss: 5.1848 | ERank W_q: 148/320
  [DPI v17.0 + SpectreMuon] Step  3500 | Loss: 5.3323 | ERank W_q: 164/320
  [DPI v17.0 + SpectreMuon] Step  4000 | Loss: 5.4249 | ERank W_q: 175/320
  [DPI v17.0 + SpectreMuon] Step  4500 | Loss: 5.1427 | ERank W_q: 186/320
  [DPI v17.0 + SpectreMuon] Step  5000 | Loss: 4.8955 | ERank W_q: 196/320
  [DPI v17.0 + SpectreMuon] Step  5500 | Loss: 5.1803 | ERank W_q: 204/320
  [DPI v17.0 + SpectreMuon] Step  6000 | Loss: 5.0701 | ERank W_q: 209/320
  [DPI v17.0 + SpectreMuon] Step  6500 | Loss: 4.8743 | ERank W_q: 215/320
  [DPI v17.0 + SpectreMuon] Step  7000 | Loss: 4.8215 | ERank W_q: 219/320
  [DPI v17.0 + SpectreMuon] Step  7500 | Loss: 4.9472 | ERank W_q: 223/320
  [DPI v17.0 + SpectreMuon] Step  8000 | Loss: 5.1318 | ERank W_q: 226/320
  [DPI v17.0 + SpectreMuon] Step  8500 | Loss: 4.8120 | ERank W_q: 231/320
  [DPI v17.0 + SpectreMuon] Step  9000 | Loss: 5.0929 | ERank W_q: 233/320
  [DPI v17.0 + SpectreMuon] Step  9500 | Loss: 5.0462 | ERank W_q: 235/320
  [DPI v17.0 + SpectreMuon] Step 10000 | Loss: 5.0533 | ERank W_q: 237/320
  [DPI v17.0 + SpectreMuon] Step 10500 | Loss: 4.8678 | ERank W_q: 239/320
  [DPI v17.0 + SpectreMuon] Step 11000 | Loss: 5.0219 | ERank W_q: 239/320
  [DPI v17.0 + SpectreMuon] Step 11500 | Loss: 4.6395 | ERank W_q: 242/320
  [DPI v17.0 + SpectreMuon] Step 12000 | Loss: 4.9895 | ERank W_q: 242/320
  [DPI v17.0 + SpectreMuon] Step 12500 | Loss: 4.9369 | ERank W_q: 243/320
  [DPI v17.0 + SpectreMuon] Step 13000 | Loss: 4.7778 | ERank W_q: 244/320
  [DPI v17.0 + SpectreMuon] Step 13500 | Loss: 4.7005 | ERank W_q: 245/320
  [DPI v17.0 + SpectreMuon] Step 14000 | Loss: 4.7772 | ERank W_q: 245/320
  [DPI v17.0 + SpectreMuon] Step 14500 | Loss: 4.8315 | ERank W_q: 245/320
  [DPI v17.0 + SpectreMuon] Step 15000 | Loss: 4.7974 | ERank W_q: 246/320
  [DPI v17.0 + SpectreMuon] Step 15500 | Loss: 4.8812 | ERank W_q: 246/320
  [DPI v17.0 + SpectreMuon] Step 16000 | Loss: 4.5870 | ERank W_q: 247/320
  [DPI v17.0 + SpectreMuon] Step 16500 | Loss: 4.8663 | ERank W_q: 248/320
  [DPI v17.0 + SpectreMuon] Step 17000 | Loss: 4.9650 | ERank W_q: 249/320
  [DPI v17.0 + SpectreMuon] Step 17500 | Loss: 4.8291 | ERank W_q: 249/320
  [DPI v17.0 + SpectreMuon] Step 18000 | Loss: 4.9887 | ERank W_q: 249/320
  [DPI v17.0 + SpectreMuon] Step 18500 | Loss: 4.6360 | ERank W_q: 249/320
  [DPI v17.0 + SpectreMuon] Step 19000 | Loss: 4.6887 | ERank W_q: 249/320
  [DPI v17.0 + SpectreMuon] Step 19500 | Loss: 4.7705 | ERank W_q: 249/320
  [DPI v17.0 + SpectreMuon] Step 20000 | Loss: 4.8114 | ERank W_q: 249/320

🚀 Running DPI v16.2 + Muon...
  [DPI] mode=V16.2 | scale=1.0000 | anchors=False
  [Phase 0] Seeding Lexical Manifold...
  [Phase 1] Semantic Clustering...
  [Phase 2] Genomic Bootstrapping...
  [Phase 4] Calibrating Output Head...
DPI COMPLETE (Standard).
  [DPI v16.2 + Muon] Step   500 | Loss: 6.4280 | ERank W_q: 297/320
  [DPI v16.2 + Muon] Step  1000 | Loss: 5.8056 | ERank W_q: 316/320
  [DPI v16.2 + Muon] Step  1500 | Loss: 5.3640 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step  2000 | Loss: 5.4855 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step  2500 | Loss: 5.5460 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step  3000 | Loss: 5.2515 | ERank W_q: 314/320
  [DPI v16.2 + Muon] Step  3500 | Loss: 5.5173 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step  4000 | Loss: 5.6161 | ERank W_q: 316/320
  [DPI v16.2 + Muon] Step  4500 | Loss: 5.3850 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step  5000 | Loss: 5.1562 | ERank W_q: 314/320
  [DPI v16.2 + Muon] Step  5500 | Loss: 5.4230 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step  6000 | Loss: 5.3864 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step  6500 | Loss: 5.2131 | ERank W_q: 316/320
  [DPI v16.2 + Muon] Step  7000 | Loss: 5.1975 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step  7500 | Loss: 5.2609 | ERank W_q: 316/320
  [DPI v16.2 + Muon] Step  8000 | Loss: 5.5096 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step  8500 | Loss: 5.2453 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step  9000 | Loss: 5.4913 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step  9500 | Loss: 5.4471 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step 10000 | Loss: 5.4404 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step 10500 | Loss: 5.2570 | ERank W_q: 316/320
  [DPI v16.2 + Muon] Step 11000 | Loss: 5.4503 | ERank W_q: 316/320
  [DPI v16.2 + Muon] Step 11500 | Loss: 5.0811 | ERank W_q: 316/320
  [DPI v16.2 + Muon] Step 12000 | Loss: 5.3989 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step 12500 | Loss: 5.3285 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step 13000 | Loss: 5.1870 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step 13500 | Loss: 5.1639 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step 14000 | Loss: 5.2207 | ERank W_q: 316/320
  [DPI v16.2 + Muon] Step 14500 | Loss: 5.2652 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step 15000 | Loss: 5.2097 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step 15500 | Loss: 5.2908 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step 16000 | Loss: 5.0605 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step 16500 | Loss: 5.3088 | ERank W_q: 314/320
  [DPI v16.2 + Muon] Step 17000 | Loss: 5.3940 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step 17500 | Loss: 5.2772 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step 18000 | Loss: 5.3807 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step 18500 | Loss: 5.0556 | ERank W_q: 315/320
  [DPI v16.2 + Muon] Step 19000 | Loss: 5.1659 | ERank W_q: 316/320
  [DPI v16.2 + Muon] Step 19500 | Loss: 5.1902 | ERank W_q: 316/320
  [DPI v16.2 + Muon] Step 20000 | Loss: 5.2266 | ERank W_q: 315/320

🚀 Running DPI v14 + Muon...
  [Phase 0] Seeding Lexical Manifold (Exact SVD)...
  [Phase 2] Sequential Bootstrapping v15.2 (Attention Arch: False)...
    Layer  0 | QK-Alignment: 0.600
    Layer  5 | QK-Alignment: 0.171
    Layer  7 | QK-Alignment: 0.000
  [Phase 3] Final Manifold Calibration...
DPI-15.2 Attention Arch Initialization Complete.
  [DPI v14 + Muon] Step   500 | Loss: 6.1937 | ERank W_q: 320/320
  [DPI v14 + Muon] Step  1000 | Loss: 5.8222 | ERank W_q: 316/320
  [DPI v14 + Muon] Step  1500 | Loss: 5.6971 | ERank W_q: 315/320
  [DPI v14 + Muon] Step  2000 | Loss: 5.4486 | ERank W_q: 315/320
  [DPI v14 + Muon] Step  2500 | Loss: 5.5102 | ERank W_q: 315/320
  [DPI v14 + Muon] Step  3000 | Loss: 5.6121 | ERank W_q: 315/320
  [DPI v14 + Muon] Step  3500 | Loss: 5.5740 | ERank W_q: 316/320
  [DPI v14 + Muon] Step  4000 | Loss: 5.4846 | ERank W_q: 315/320
  [DPI v14 + Muon] Step  4500 | Loss: 5.6734 | ERank W_q: 315/320
  [DPI v14 + Muon] Step  5000 | Loss: 5.7223 | ERank W_q: 315/320
  [DPI v14 + Muon] Step  5500 | Loss: 5.5724 | ERank W_q: 315/320
  [DPI v14 + Muon] Step  6000 | Loss: 5.4905 | ERank W_q: 316/320
  [DPI v14 + Muon] Step  6500 | Loss: 5.3809 | ERank W_q: 314/320
  [DPI v14 + Muon] Step  7000 | Loss: 5.4589 | ERank W_q: 314/320
  [DPI v14 + Muon] Step  7500 | Loss: 5.4309 | ERank W_q: 315/320
  [DPI v14 + Muon] Step  8000 | Loss: 5.4844 | ERank W_q: 315/320
  [DPI v14 + Muon] Step  8500 | Loss: 5.6185 | ERank W_q: 315/320
  [DPI v14 + Muon] Step  9000 | Loss: 5.2903 | ERank W_q: 316/320
  [DPI v14 + Muon] Step  9500 | Loss: 5.2736 | ERank W_q: 316/320
  [DPI v14 + Muon] Step 10000 | Loss: 5.5847 | ERank W_q: 315/320
  [DPI v14 + Muon] Step 10500 | Loss: 5.4567 | ERank W_q: 315/320
  [DPI v14 + Muon] Step 11000 | Loss: 5.4659 | ERank W_q: 316/320
  [DPI v14 + Muon] Step 11500 | Loss: 5.2898 | ERank W_q: 315/320
  [DPI v14 + Muon] Step 12000 | Loss: 5.4586 | ERank W_q: 315/320
  [DPI v14 + Muon] Step 12500 | Loss: 5.5725 | ERank W_q: 314/320
  [DPI v14 + Muon] Step 13000 | Loss: 5.3386 | ERank W_q: 315/320
  [DPI v14 + Muon] Step 13500 | Loss: 5.3895 | ERank W_q: 315/320
  [DPI v14 + Muon] Step 14000 | Loss: 5.1965 | ERank W_q: 315/320
  [DPI v14 + Muon] Step 14500 | Loss: 5.4406 | ERank W_q: 314/320
  [DPI v14 + Muon] Step 15000 | Loss: 5.3499 | ERank W_q: 314/320
  [DPI v14 + Muon] Step 15500 | Loss: 5.2942 | ERank W_q: 315/320
  [DPI v14 + Muon] Step 16000 | Loss: 5.2614 | ERank W_q: 315/320
  [DPI v14 + Muon] Step 16500 | Loss: 5.0608 | ERank W_q: 315/320
  [DPI v14 + Muon] Step 17000 | Loss: 5.4128 | ERank W_q: 315/320
  [DPI v14 + Muon] Step 17500 | Loss: 5.4124 | ERank W_q: 315/320
  [DPI v14 + Muon] Step 18000 | Loss: 5.4821 | ERank W_q: 315/320
  [DPI v14 + Muon] Step 18500 | Loss: 5.3834 | ERank W_q: 314/320
  [DPI v14 + Muon] Step 19000 | Loss: 5.3106 | ERank W_q: 314/320
  [DPI v14 + Muon] Step 19500 | Loss: 5.5148 | ERank W_q: 315/320
  [DPI v14 + Muon] Step 20000 | Loss: 5.1257 | ERank W_q: 315/320

