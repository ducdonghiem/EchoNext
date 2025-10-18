import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import warnings
import pywt

warnings.filterwarnings('ignore')

DEFAULT_FS = 250

class FeatureExtraction:
    """
    Clean ECG feature extraction following distinct categories:
    1. HRV - Time Domain
    2. HRV - Geometric 
    3. HRV - Frequency Domain
    4. HRV - Nonlinear
    5. Morphological (P-QRS-T)
    6. Signal-Based (Statistical & Spectral)
    7. Beat-to-Beat Dynamics
    """
    
    def __init__(self, sampling_rate=DEFAULT_FS):
        self.fs = sampling_rate
        self._cache = {}
    
    def _get_rr_data(self, ecg_signal):
        """
        Detect R-peaks and compute RR intervals (in ms).
        Cache to avoid recomputation.
        """
        if 'rr_data' in self._cache:
            return self._cache['rr_data']
        
        signal = pd.Series(ecg_signal)
        cleaned_ecg = nk.ecg_clean(signal, sampling_rate=self.fs)
        _, r_peaks = nk.ecg_peaks(cleaned_ecg, sampling_rate=self.fs)
        r_peaks_indices = r_peaks['ECG_R_Peaks']
        
        if len(r_peaks_indices) < 2:
            raise ValueError(f"Insufficient R-peaks detected: {len(r_peaks_indices)}")
        
        # RR intervals in milliseconds
        rr_intervals = np.diff(r_peaks_indices) / self.fs * 1000
        
        self._cache['rr_data'] = {
            'rr_intervals': rr_intervals,
            'r_peaks': r_peaks_indices,
            'cleaned_ecg': cleaned_ecg
        }
        return self._cache['rr_data']
    
    # =========================================================================
    # CATEGORY 1: HRV - TIME DOMAIN
    # =========================================================================
    def extract_hrv_time_domain(self, ecg_signal):
        """
        Time-domain HRV metrics from RR intervals.
        Features: MeanNN, SDNN, RMSSD, SDSD, pNN50, pNN20, HR
        """
        rr = self._get_rr_data(ecg_signal)['rr_intervals']
        
        features = {}
        
        # Basic statistics
        features['HRV_MeanNN'] = np.mean(rr)
        features['HRV_SDNN'] = np.std(rr, ddof=1)
        features['HRV_MedianNN'] = np.median(rr)
        features['HRV_CVNN'] = (features['HRV_SDNN'] / features['HRV_MeanNN']) * 100
        
        # Difference-based metrics
        diff_rr = np.diff(rr)
        features['HRV_RMSSD'] = np.sqrt(np.mean(diff_rr ** 2))
        features['HRV_SDSD'] = np.std(diff_rr, ddof=1)
        
        # pNNxx: percentage of successive RR differences > xx ms
        features['HRV_pNN50'] = (np.sum(np.abs(diff_rr) > 50) / len(diff_rr)) * 100
        features['HRV_pNN20'] = (np.sum(np.abs(diff_rr) > 20) / len(diff_rr)) * 100
        
        # Heart rate (BPM)
        features['HRV_MeanHR'] = 60000 / features['HRV_MeanNN']
        
        return pd.Series(features)
    
    # =========================================================================
    # CATEGORY 2: HRV - GEOMETRIC
    # =========================================================================
    def extract_hrv_geometric(self, ecg_signal):
        """
        Geometric HRV metrics from RR interval histogram.
        Features: Triangular Index (TRI), TINN
        """
        rr = self._get_rr_data(ecg_signal)['rr_intervals']
        
        features = {}
        
        # Triangular Index: total number of RR intervals / height of histogram
        hist, bin_edges = np.histogram(rr, bins=np.arange(rr.min(), rr.max() + 8, 8))
        features['HRV_TriangularIndex'] = len(rr) / np.max(hist) if np.max(hist) > 0 else np.nan
        
        # TINN: Triangular Interpolation of NN interval histogram
        # Base width of the distribution measured as a base of a triangle
        # approximating the NN interval distribution
        if len(hist) > 2:
            max_idx = np.argmax(hist)
            # Find left and right boundaries where histogram drops significantly
            left = max_idx
            while left > 0 and hist[left - 1] > hist[max_idx] * 0.05:
                left -= 1
            right = max_idx
            while right < len(hist) - 1 and hist[right + 1] > hist[max_idx] * 0.05:
                right += 1
            features['HRV_TINN'] = (right - left) * 8  # bin width is 8ms
        else:
            # print("Insufficient histogram bins for TINN calculation.")
            features['HRV_TINN'] = np.nan
        
        # print(features['HRV_TINN'])
        return pd.Series(features)
    
    # =========================================================================
    # CATEGORY 3: HRV - FREQUENCY DOMAIN
    # =========================================================================
    def extract_hrv_frequency_domain(self, ecg_signal):
        """
        Frequency-domain HRV metrics using Lomb-Scargle periodogram.
        Features: VLF, LF, HF power, LF/HF ratio, normalized LF and HF
        """
        rr_data = self._get_rr_data(ecg_signal)
        rr = rr_data['rr_intervals']
        r_peaks = rr_data['r_peaks']
        
        features = {}
        
        # Time stamps of RR intervals (in seconds)
        rr_times = r_peaks[:-1] / self.fs
        
        # Interpolate RR intervals to get evenly sampled signal
        from scipy.interpolate import interp1d
        interp_func = interp1d(rr_times, rr, kind='cubic', fill_value='extrapolate')
        
        # Create uniform time grid at 4 Hz (standard for HRV analysis)
        t_uniform = np.arange(rr_times[0], rr_times[-1], 0.25)
        rr_uniform = interp_func(t_uniform)
        
        # Compute PSD using Welch's method
        freqs, psd = welch(rr_uniform, fs=4, nperseg=min(256, len(rr_uniform)))
        
        # Define frequency bands (Hz)
        vlf_band = (0.003, 0.04)
        lf_band = (0.04, 0.15)
        hf_band = (0.15, 0.4)
        
        # Calculate power in each band
        vlf_power = np.trapezoid(psd[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])], 
                                  freqs[(freqs >= vlf_band[0]) & (freqs < vlf_band[1])])
        lf_power = np.trapezoid(psd[(freqs >= lf_band[0]) & (freqs < lf_band[1])], 
                                 freqs[(freqs >= lf_band[0]) & (freqs < lf_band[1])])
        hf_power = np.trapezoid(psd[(freqs >= hf_band[0]) & (freqs < hf_band[1])], 
                                 freqs[(freqs >= hf_band[0]) & (freqs < hf_band[1])])
        
        total_power = vlf_power + lf_power + hf_power
        
        features['HRV_VLF'] = vlf_power
        features['HRV_LF'] = lf_power
        features['HRV_HF'] = hf_power
        features['HRV_TotalPower'] = total_power
        features['HRV_LF_HF_Ratio'] = lf_power / hf_power if hf_power > 0 else np.nan
        
        # Normalized powers
        lf_hf_sum = lf_power + hf_power
        features['HRV_LFn'] = (lf_power / lf_hf_sum) * 100 if lf_hf_sum > 0 else np.nan
        features['HRV_HFn'] = (hf_power / lf_hf_sum) * 100 if lf_hf_sum > 0 else np.nan
        
        return pd.Series(features)
    
    # =========================================================================
    # CATEGORY 3.5: HRV - FREQUENCY DOMAIN
    # =========================================================================
    def extract_hrv_wavelet(self, ecg_signal):
        """
        Wavelet-based HRV features.
        Features: Energy of each wavelet band (D1–D4), total energy, normalized energies,
                and wavelet entropy.
        """
        rr_data = self._get_rr_data(ecg_signal)
        rr = rr_data['rr_intervals']
        r_peaks = rr_data['r_peaks']

        # Interpolate RR intervals (uniform sampling)
        from scipy.interpolate import interp1d
        rr_times = r_peaks[:-1] / self.fs
        interp_func = interp1d(rr_times, rr, kind='cubic', fill_value='extrapolate')
        t_uniform = np.arange(rr_times[0], rr_times[-1], 0.25)
        rr_uniform = interp_func(t_uniform)

        # Wavelet decomposition (Daubechies 4 is commonly used)
        coeffs = pywt.wavedec(rr_uniform, 'db4', level=4)
        features = {}

        # Compute energy in each wavelet detail
        energies = [np.sum(np.square(c)) for c in coeffs]
        total_energy = np.sum(energies)

        for i, e in enumerate(energies):
            features[f'HRV_Wavelet_Energy_D{i+1}'] = e
            features[f'HRV_Wavelet_EnergyNorm_D{i+1}'] = (e / total_energy) * 100 if total_energy > 0 else np.nan

        features['HRV_Wavelet_TotalEnergy'] = total_energy

        # Wavelet entropy (measure of signal complexity)
        probs = np.array(energies) / total_energy if total_energy > 0 else np.zeros_like(energies)
        features['HRV_Wavelet_Entropy'] = -np.sum(probs * np.log2(probs + 1e-12))

        return pd.Series(features)
    
    # =========================================================================
    # CATEGORY 4: HRV - NONLINEAR
    # =========================================================================
    def extract_hrv_nonlinear(self, ecg_signal):
        """
        Nonlinear HRV metrics.
        Features: SD1, SD2, SD1/SD2 (Poincaré), Sample Entropy, DFA α1
        """
        rr = self._get_rr_data(ecg_signal)['rr_intervals']
        
        features = {}
        
        # Poincaré plot features (SD1, SD2)
        # SD1: standard deviation perpendicular to line of identity
        # SD2: standard deviation along line of identity
        diff_rr = np.diff(rr)
        features['HRV_SD1'] = np.sqrt(np.var(diff_rr) / 2)
        features['HRV_SD2'] = np.sqrt(2 * np.var(rr) - np.var(diff_rr) / 2)
        features['HRV_SD1_SD2_Ratio'] = features['HRV_SD1'] / features['HRV_SD2'] if features['HRV_SD2'] > 0 else np.nan
        
        # Sample Entropy - extract only the numerical value
        sampen_result = nk.entropy_sample(rr, dimension=2, tolerance=0.2 * np.std(rr))
        features['HRV_SampEn'] = sampen_result[0] if isinstance(sampen_result, tuple) else sampen_result
        
        # Approximate Entropy - extract only the numerical value
        apen_result = nk.entropy_approximate(rr, dimension=2, tolerance=0.2 * np.std(rr))
        features['HRV_ApEn'] = apen_result[0] if isinstance(apen_result, tuple) else apen_result
        
        # Detrended Fluctuation Analysis (DFA) - short-term scaling exponent
        # Scale must be adjusted based on RR interval length
        max_scale = len(rr) // 4  # Conservative: at least 4 windows
        if max_scale >= 4:
            scale = [4, min(16, max_scale)]
            dfa_result = nk.fractal_dfa(rr, scale=scale)
            if isinstance(dfa_result, tuple):
                features['HRV_DFA_alpha1'] = dfa_result[0]
            else:
                features['HRV_DFA_alpha1'] = dfa_result[0] if len(dfa_result) > 0 else np.nan
        else:
            # Not enough data for DFA
            features['HRV_DFA_alpha1'] = np.nan
        
        # Correlation Dimension - extract only the numerical value
        cd_result = nk.complexity_cd(rr)
        features['HRV_CorrelationDim'] = cd_result[0] if isinstance(cd_result, tuple) else cd_result
        
        return pd.Series(features)
    
    # =========================================================================
    # CATEGORY 5: MORPHOLOGICAL FEATURES (P-QRS-T)
    # =========================================================================
    def extract_morphological_features(self, ecg_signal):
        """
        Morphological features from ECG waveform (P, QRS, T waves).
        Features: Amplitudes, Durations, Intervals (PR, QRS, QT, QTc)
        """
        rr_data = self._get_rr_data(ecg_signal)
        r_peaks = rr_data['r_peaks']
        cleaned_ecg = rr_data['cleaned_ecg']
        rr_intervals = rr_data['rr_intervals']
        
        # Delineate ECG waves
        _, waves = nk.ecg_delineate(cleaned_ecg, r_peaks, sampling_rate=self.fs, 
                                     show=False, method="dwt")
        
        features = {}
        ecg_array = np.array(cleaned_ecg)
        
        # --- AMPLITUDES ---
        for wave_name in ['P', 'Q', 'R', 'S', 'T']:
            key = f'ECG_{wave_name}_Peaks'
            if key in waves:
                peaks = np.array(waves[key])
                valid = ~np.isnan(peaks)
                if np.any(valid):
                    indices = peaks[valid].astype(int)
                    indices = indices[(indices >= 0) & (indices < len(ecg_array))]
                    if len(indices) > 0:
                        amps = ecg_array[indices]
                        features[f'Morph_{wave_name}_Amp_Mean'] = np.mean(amps)
                        features[f'Morph_{wave_name}_Amp_Std'] = np.std(amps)
        
        # --- DURATIONS ---
        # P wave duration
        if 'ECG_P_Onsets' in waves and 'ECG_P_Offsets' in waves:
            p_on = np.array(waves['ECG_P_Onsets'])
            p_off = np.array(waves['ECG_P_Offsets'])
            valid = (~np.isnan(p_on)) & (~np.isnan(p_off))
            if np.any(valid):
                p_dur = (p_off[valid] - p_on[valid]) / self.fs * 1000
                features['Morph_P_Duration_Mean'] = np.mean(p_dur)
                features['Morph_P_Duration_Std'] = np.std(p_dur)
        
        # QRS duration
        if 'ECG_Q_Peaks' in waves and 'ECG_S_Peaks' in waves:
            q_peaks = np.array(waves['ECG_Q_Peaks'])
            s_peaks = np.array(waves['ECG_S_Peaks'])
            valid = (~np.isnan(q_peaks)) & (~np.isnan(s_peaks))
            if np.any(valid):
                qrs_dur = (s_peaks[valid] - q_peaks[valid]) / self.fs * 1000
                features['Morph_QRS_Duration_Mean'] = np.mean(qrs_dur)
                features['Morph_QRS_Duration_Std'] = np.std(qrs_dur)
        
        # T wave duration
        if 'ECG_T_Onsets' in waves and 'ECG_T_Offsets' in waves:
            t_on = np.array(waves['ECG_T_Onsets'])
            t_off = np.array(waves['ECG_T_Offsets'])
            valid = (~np.isnan(t_on)) & (~np.isnan(t_off))
            if np.any(valid):
                t_dur = (t_off[valid] - t_on[valid]) / self.fs * 1000
                features['Morph_T_Duration_Mean'] = np.mean(t_dur)
                features['Morph_T_Duration_Std'] = np.std(t_dur)
        
        # --- INTERVALS ---
        # PR interval (P onset to R onset)
        if 'ECG_P_Onsets' in waves and 'ECG_R_Onsets' in waves:
            p_on = np.array(waves['ECG_P_Onsets'])
            r_on = np.array(waves['ECG_R_Onsets'])
            valid = (~np.isnan(p_on)) & (~np.isnan(r_on))
            if np.any(valid):
                pr_int = (r_on[valid] - p_on[valid]) / self.fs * 1000
                features['Morph_PR_Interval_Mean'] = np.mean(pr_int)
                features['Morph_PR_Interval_Std'] = np.std(pr_int)
        
        # QT interval (R onset to T offset) and QTc (Bazett's correction)
        if 'ECG_R_Onsets' in waves and 'ECG_T_Offsets' in waves:
            r_on = np.array(waves['ECG_R_Onsets'])
            t_off = np.array(waves['ECG_T_Offsets'])
            valid = (~np.isnan(r_on)) & (~np.isnan(t_off))
            if np.any(valid):
                qt_int = (t_off[valid] - r_on[valid]) / self.fs * 1000
                features['Morph_QT_Interval_Mean'] = np.mean(qt_int)
                features['Morph_QT_Interval_Std'] = np.std(qt_int)
                
                # QTc = QT / sqrt(RR in seconds)
                mean_rr_sec = np.mean(rr_intervals) / 1000
                qtc = np.mean(qt_int) / np.sqrt(mean_rr_sec)
                features['Morph_QTc_Bazett'] = qtc
        
        # --- AMPLITUDE RATIOS ---
        if 'Morph_T_Amp_Mean' in features and 'Morph_R_Amp_Mean' in features:
            features['Morph_T_R_Ratio'] = features['Morph_T_Amp_Mean'] / features['Morph_R_Amp_Mean']
        
        return pd.Series(features)
    
    # =========================================================================
    # CATEGORY 6: SIGNAL-BASED (STATISTICAL & SPECTRAL)
    # =========================================================================
    def extract_signal_features(self, ecg_signal):
        """
        Statistical and spectral features from the entire ECG signal.
        Features: Mean, Std, Skewness, Kurtosis, RMS, Energy, Spectral bands
        """
        signal = ecg_signal.copy()
        features = {}
        
        # --- STATISTICAL FEATURES ---
        features['Signal_Mean'] = np.mean(signal)
        features['Signal_Std'] = np.std(signal)
        features['Signal_Median'] = np.median(signal)
        features['Signal_Min'] = np.min(signal)
        features['Signal_Max'] = np.max(signal)
        features['Signal_Range'] = features['Signal_Max'] - features['Signal_Min']
        features['Signal_Skewness'] = skew(signal)
        features['Signal_Kurtosis'] = kurtosis(signal)
        features['Signal_RMS'] = np.sqrt(np.mean(signal ** 2))
        features['Signal_Energy'] = np.sum(signal ** 2)
        
        # Zero-crossing rate
        features['Signal_ZeroCrossing'] = np.sum(np.diff(np.sign(signal)) != 0)
        
        # --- SPECTRAL FEATURES ---
        freqs, psd = welch(signal, fs=self.fs, nperseg=min(256, len(signal)))
        
        # Define frequency bands for ECG
        bands = {
            'VLF': (0, 1),      # Very low frequency noise
            'LF': (1, 10),      # Low frequency content
            'MF': (10, 40),     # Mid frequency (main ECG content)
            'HF': (40, 100)     # High frequency
        }
        
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            if np.any(mask):
                features[f'Signal_Power_{band_name}'] = np.trapezoid(psd[mask], freqs[mask])
        
        # Total spectral power
        features['Signal_Power_Total'] = np.trapezoid(psd, freqs)
        
        # Dominant frequency
        features['Signal_DominantFreq'] = freqs[np.argmax(psd)]
        
        # Spectral entropy
        psd_norm = psd / np.sum(psd)
        psd_norm = psd_norm[psd_norm > 0]
        features['Signal_SpectralEntropy'] = -np.sum(psd_norm * np.log2(psd_norm))
        
        return pd.Series(features)
    
    # =========================================================================
    # CATEGORY 7: BEAT-TO-BEAT DYNAMICS
    # =========================================================================
    def extract_beat_to_beat_features(self, ecg_signal):
        """
        Beat-to-beat variability and dynamics.
        Features: RR ratio patterns, R-amplitude variability
        """
        rr_data = self._get_rr_data(ecg_signal)
        rr = rr_data['rr_intervals']
        r_peaks = rr_data['r_peaks']
        cleaned_ecg = rr_data['cleaned_ecg']
        
        features = {}
        
        # RR interval ratios (consecutive)
        rr_ratios = rr[1:] / rr[:-1]
        features['BtB_RR_Ratio_Mean'] = np.mean(rr_ratios)
        features['BtB_RR_Ratio_Std'] = np.std(rr_ratios)
        features['BtB_RR_Ratio_Range'] = np.max(rr_ratios) - np.min(rr_ratios)
        
        # R-peak amplitude variability
        ecg_array = np.array(cleaned_ecg)
        r_amps = ecg_array[r_peaks]
        features['BtB_R_Amp_Mean'] = np.mean(r_amps)
        features['BtB_R_Amp_Std'] = np.std(r_amps)
        features['BtB_R_Amp_CV'] = features['BtB_R_Amp_Std'] / features['BtB_R_Amp_Mean'] * 100
        
        # R-amplitude differences
        r_amp_diffs = np.diff(r_amps)
        features['BtB_R_Amp_Diff_Mean'] = np.mean(np.abs(r_amp_diffs))
        features['BtB_R_Amp_Diff_Std'] = np.std(r_amp_diffs)
        
        # Correlation between consecutive RR intervals
        if len(rr) > 1:
            features['BtB_RR_Autocorr'] = np.corrcoef(rr[:-1], rr[1:])[0, 1]
        
        return pd.Series(features)
    
    # =========================================================================
    # EXTRACT ALL FEATURES
    # =========================================================================
    def extract_all_features(self, ecg_signal):
        """
        Extract all feature categories and combine into a single Series.
        """
        self._cache = {}  # Clear cache
        
        # Extract each category
        cat1 = self.extract_hrv_time_domain(ecg_signal)
        cat2 = self.extract_hrv_geometric(ecg_signal)
        cat3 = self.extract_hrv_frequency_domain(ecg_signal)
        cat35 = self.extract_hrv_wavelet(ecg_signal)
        cat4 = self.extract_hrv_nonlinear(ecg_signal)
        cat5 = self.extract_morphological_features(ecg_signal)
        cat6 = self.extract_signal_features(ecg_signal)
        cat7 = self.extract_beat_to_beat_features(ecg_signal)
        
        # Combine all features
        all_features = pd.concat([cat1, cat2, cat3, cat35, cat4, cat5, cat6, cat7])
        
        # Check for NaN values
        nan_features = all_features[all_features.isna()]
        if len(nan_features) > 0:
            print(f"WARNING: {len(nan_features)} features contain NaN values:")
            print(nan_features.index.tolist())
        
        return all_features


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == '__main__':
    fs = 250
    
    # Load ECG signal (Lead II)
    ecg = np.load('1.1.0/EchoNext_train_waveforms.npy')[1, 0, :, 1]
    
    print(f"ECG signal loaded: {ecg.shape}")
    
    extractor = FeatureExtraction(sampling_rate=fs)
    
    # Extract all features
    features = extractor.extract_all_features(ecg)
    
    # Save to CSV
    df = pd.DataFrame(features).transpose()
    df.to_csv('new_extracted_ecg_features.csv', index=False)
    
    # Display summary
    print(f"\n{'='*60}")
    print(f"FEATURE EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total features extracted: {len(features)}")
    print(f"\nFeature breakdown by category:")
    
    categories = {
        'HRV Time': [k for k in features.index if k.startswith('HRV_') and not any(x in k for x in ['VLF', 'LF', 'HF', 'SD1', 'SD2', 'SampEn', 'ApEn', 'DFA', 'Correlation', 'Triangular', 'TINN'])],
        'HRV Geometric': [k for k in features.index if 'Triangular' in k or 'TINN' in k],
        'HRV Frequency': [k for k in features.index if any(x in k for x in ['VLF', 'LF', 'HF', 'TotalPower'])],
        'HRV Wavelet': [k for k in features.index if 'Wavelet' in k],
        'HRV Nonlinear': [k for k in features.index if any(x in k for x in ['SD1', 'SD2', 'SampEn', 'ApEn', 'DFA', 'Correlation'])],
        'Morphological': [k for k in features.index if k.startswith('Morph_')],
        'Signal-Based': [k for k in features.index if k.startswith('Signal_')],
        'Beat-to-Beat': [k for k in features.index if k.startswith('BtB_')]
    }
    
    for cat_name, cat_features in categories.items():
        print(f"  {cat_name}: {len(cat_features)} features")
    
    print(f"\n{'='*60}")
    print(f"First 10 features:")
    print(features.head(10).to_string())
    print(f"\n{'='*60}\n")