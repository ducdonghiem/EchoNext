# Feature Selection Analysis Report

**Generated:** 2025-10-24 20:23:12

**Total Targets Analyzed:** 12

## Summary Statistics

| Target                                           |   N_Samples |   Positive_Rate_% |   Mean_Top_F_Score |   Max_F_Score |   Min_F_Score |
|:-------------------------------------------------|------------:|------------------:|-------------------:|--------------:|--------------:|
| lvef_lte_45_flag                                 |       72378 |             23.35 |            2577.32 |       4330.21 |       1666.90 |
| lvwt_gte_13_flag                                 |       72378 |             24.36 |             697.98 |        986.47 |        403.72 |
| aortic_stenosis_moderate_or_greater_flag         |       72378 |              4.03 |             343.96 |        404.22 |        257.24 |
| aortic_regurgitation_moderate_or_greater_flag    |       72378 |              1.21 |              29.03 |         42.74 |         20.78 |
| mitral_regurgitation_moderate_or_greater_flag    |       72378 |              8.46 |             718.35 |        882.65 |        582.54 |
| tricuspid_regurgitation_moderate_or_greater_flag |       72378 |             10.62 |            1066.19 |       1452.71 |        831.27 |
| pulmonary_regurgitation_moderate_or_greater_flag |       72378 |              0.83 |              79.86 |        112.48 |         55.95 |
| rv_systolic_dysfunction_moderate_or_greater_flag |       72378 |             13.21 |            1780.04 |       2799.22 |       1150.68 |
| pericardial_effusion_moderate_large_flag         |       72378 |              2.86 |             268.44 |        347.89 |        196.61 |
| pasp_gte_45_flag                                 |       72378 |             18.93 |             828.17 |       1030.02 |        547.49 |
| tr_max_gte_32_flag                               |       72378 |             10.34 |             463.58 |        589.12 |        309.43 |
| shd_moderate_or_greater_flag                     |       72378 |             52.33 |            2595.40 |       3963.59 |       1658.23 |

## Most Frequently Selected Features

| Feature                 |   Frequency |   Percentage |
|:------------------------|------------:|-------------:|
| Signal_Mean             |          11 |        91.67 |
| Morph_T_Amp_Mean        |          11 |        91.67 |
| Morph_Q_Amp_Mean        |           9 |        75.00 |
| Morph_QT_Interval_Std   |           8 |        66.67 |
| Signal_SpectralEntropy  |           7 |        58.33 |
| Morph_T_Duration_Mean   |           7 |        58.33 |
| Signal_Skewness         |           6 |        50.00 |
| Signal_Max              |           6 |        50.00 |
| BtB_RR_Autocorr         |           6 |        50.00 |
| HRV_pNN50               |           6 |        50.00 |
| Morph_P_Duration_Std    |           5 |        41.67 |
| BtB_RR_Ratio_Range      |           5 |        41.67 |
| BtB_R_Amp_Mean          |           3 |        25.00 |
| Signal_Kurtosis         |           3 |        25.00 |
| Signal_Min              |           3 |        25.00 |
| Morph_S_Amp_Mean        |           3 |        25.00 |
| Morph_QT_Interval_Mean  |           3 |        25.00 |
| Morph_QRS_Duration_Mean |           2 |        16.67 |
| Morph_PR_Interval_Mean  |           2 |        16.67 |
| Morph_PR_Interval_Std   |           2 |        16.67 |

## Top Features Per Target

### lvef_lte_45_flag

- **Positive Rate:** 23.35%
- **Mean F-Score (Top 10):** 2577.32
- **Top Features:** Signal_Mean, Signal_SpectralEntropy, Signal_Skewness, Morph_T_Amp_Mean, Signal_Max

### lvwt_gte_13_flag

- **Positive Rate:** 24.36%
- **Mean F-Score (Top 10):** 697.98
- **Top Features:** Signal_Skewness, Signal_Min, Signal_Mean, Morph_S_Amp_Mean, Signal_SpectralEntropy

### aortic_stenosis_moderate_or_greater_flag

- **Positive Rate:** 4.03%
- **Mean F-Score (Top 10):** 343.96
- **Top Features:** HRV_pNN50, Morph_Q_Amp_Mean, Morph_PR_Interval_Std, HRV_pNN20, Morph_PR_Interval_Mean

### aortic_regurgitation_moderate_or_greater_flag

- **Positive Rate:** 1.21%
- **Mean F-Score (Top 10):** 29.03
- **Top Features:** Signal_Min, Morph_P_Duration_Std, Morph_S_Amp_Mean, Morph_T_Amp_Mean, HRV_pNN50

### mitral_regurgitation_moderate_or_greater_flag

- **Positive Rate:** 8.46%
- **Mean F-Score (Top 10):** 718.35
- **Top Features:** Signal_SpectralEntropy, HRV_pNN50, Signal_Mean, Morph_Q_Amp_Mean, HRV_CVNN

### tricuspid_regurgitation_moderate_or_greater_flag

- **Positive Rate:** 10.62%
- **Mean F-Score (Top 10):** 1066.19
- **Top Features:** HRV_pNN50, Signal_Mean, Morph_Q_Amp_Mean, HRV_CVNN, BtB_RR_Ratio_Range

### pulmonary_regurgitation_moderate_or_greater_flag

- **Positive Rate:** 0.83%
- **Mean F-Score (Top 10):** 79.86
- **Top Features:** Signal_Mean, Signal_Skewness, Morph_QT_Interval_Std, BtB_RR_Autocorr, Morph_T_Amp_Mean

### rv_systolic_dysfunction_moderate_or_greater_flag

- **Positive Rate:** 13.21%
- **Mean F-Score (Top 10):** 1780.04
- **Top Features:** Signal_Mean, Signal_Skewness, Signal_Max, BtB_R_Amp_Mean, Morph_T_Duration_Mean

### pericardial_effusion_moderate_large_flag

- **Positive Rate:** 2.86%
- **Mean F-Score (Top 10):** 268.44
- **Top Features:** Morph_QT_Interval_Mean, Signal_Range, Signal_Max, BtB_R_Amp_Mean, HRV_MeanHR

### pasp_gte_45_flag

- **Positive Rate:** 18.93%
- **Mean F-Score (Top 10):** 828.17
- **Top Features:** Morph_Q_Amp_Mean, BtB_RR_Autocorr, Morph_T_Amp_Mean, Morph_QT_Interval_Std, Morph_T_Duration_Mean

### tr_max_gte_32_flag

- **Positive Rate:** 10.34%
- **Mean F-Score (Top 10):** 463.58
- **Top Features:** Morph_Q_Amp_Mean, BtB_RR_Autocorr, Morph_T_Amp_Mean, Morph_T_Duration_Mean, Signal_Mean

### shd_moderate_or_greater_flag

- **Positive Rate:** 52.33%
- **Mean F-Score (Top 10):** 2595.40
- **Top Features:** Signal_Mean, Morph_T_Amp_Mean, Signal_Skewness, Morph_QT_Interval_Std, Signal_SpectralEntropy

