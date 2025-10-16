import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import warnings

# The standard sampling rate for the EchoNext dataset is 250 Hz (10s segment).
DEFAULT_FS = 250

class FeatureExtraction:
    """
    Enhanced class for extracting various categories of features from a single-lead ECG signal.
    
    The methods in this class are designed to be run on a 1-D numpy array representing 
    a single-lead ECG segment (e.g., Lead II, or any lead chosen by the caller).
    
    Dependencies: numpy, pandas, neurokit2, scipy
    """
    def __init__(self, sampling_rate=DEFAULT_FS):
        """
        Initializes the FeatureExtraction class.
        
        Args:
            sampling_rate (int): The sampling rate of the ECG signal in Hz.
        """
        self.fs = sampling_rate
        self._rr_data_cache = {}

    def _calculate_rr_data(self, ecg_signal):
        """
        Internal utility function to detect R-peaks and compute RR intervals.
        Caches the result to prevent redundant computation across feature methods.

        Args:
            ecg_signal (np.ndarray): The 1-D ECG signal.

        Returns:
            dict: Contains 'rr_intervals' (list of ms), 'r_peaks' (list of indices), 
                  'cleaned_ecg', and 'heart_rate'.
        """
        if len(self._rr_data_cache) > 0:
            return self._rr_data_cache

        signal = pd.Series(ecg_signal)

        try:
            cleaned_ecg = nk.ecg_clean(signal, sampling_rate=self.fs)
            _, r_peaks = nk.ecg_peaks(cleaned_ecg, sampling_rate=self.fs)
            r_peaks_indices = r_peaks['ECG_R_Peaks']

            if len(r_peaks_indices) > 1:
                rr_intervals = (np.diff(r_peaks_indices) / self.fs) * 1000
                mean_rr = np.mean(rr_intervals)
                heart_rate = 60000 / mean_rr if mean_rr > 0 else np.nan
            else:
                rr_intervals = np.array([])
                heart_rate = np.nan

            self._rr_data_cache = {
                'rr_intervals': rr_intervals,
                'r_peaks': r_peaks_indices,
                'cleaned_ecg': cleaned_ecg,
                'heart_rate': heart_rate
            }
            return self._rr_data_cache

        except Exception as e:
            print(f"Error during R-peak detection: {e}")
            return {
                'rr_intervals': np.array([]), 
                'r_peaks': np.array([]), 
                'cleaned_ecg': signal,
                'heart_rate': np.nan
            }

    def extract_basic_features(self, ecg_signal):
        """
        Extracts basic heart rate features.
        
        Args:
            ecg_signal (np.ndarray): The 1-D ECG signal.
            
        Returns:
            pd.Series: Basic features including heart rate and RR statistics.
        """
        rr_data = self._calculate_rr_data(ecg_signal)
        rr_intervals = rr_data['rr_intervals']
        
        features = {
            'Basic_Heart_Rate_BPM': rr_data['heart_rate'],
            'Basic_Num_RR_Intervals': len(rr_intervals)
        }
        
        if len(rr_intervals) > 0:
            features['Basic_Mean_RR_ms'] = np.mean(rr_intervals)
            features['Basic_Std_RR_ms'] = np.std(rr_intervals)
            features['Basic_Min_RR_ms'] = np.min(rr_intervals)
            features['Basic_Max_RR_ms'] = np.max(rr_intervals)
        
        return pd.Series(features)

    def extract_hrv_features(self, ecg_signal):
        """
        Extracts comprehensive Heart Rate Variability (HRV) features.
        
        Args:
            ecg_signal (np.ndarray): The 1-D ECG signal.

        Returns:
            pd.Series: A Series of HRV features (Time, Frequency, Non-Linear).
        """
        rr_data = self._calculate_rr_data(ecg_signal)
        r_peaks_indices = rr_data['r_peaks']
        rr_intervals = rr_data['rr_intervals']

        if len(r_peaks_indices) < 5:
            print("Warning: Too few R-peaks for HRV calculation.")
            return pd.Series({})

        hrv_dict = {}
        
        # Time domain features (manual calculation for robustness)
        if len(rr_intervals) > 1:
            hrv_dict['HRV_MeanNN'] = np.mean(rr_intervals)
            hrv_dict['HRV_SDNN'] = np.std(rr_intervals, ddof=1)
            hrv_dict['HRV_MedianNN'] = np.median(rr_intervals)
            hrv_dict['HRV_MadNN'] = np.median(np.abs(rr_intervals - np.median(rr_intervals)))
            hrv_dict['HRV_CVNN'] = (hrv_dict['HRV_SDNN'] / hrv_dict['HRV_MeanNN']) * 100 if hrv_dict['HRV_MeanNN'] > 0 else np.nan
            
            # Difference-based features
            if len(rr_intervals) > 2:
                diff_rr = np.diff(rr_intervals)
                hrv_dict['HRV_RMSSD'] = np.sqrt(np.mean(diff_rr ** 2))
                hrv_dict['HRV_SDSD'] = np.std(diff_rr, ddof=1)
                hrv_dict['HRV_pNN50'] = (np.sum(np.abs(diff_rr) > 50) / len(diff_rr)) * 100
                hrv_dict['HRV_pNN20'] = (np.sum(np.abs(diff_rr) > 20) / len(diff_rr)) * 100

        # Use NeuroKit2 for comprehensive HRV analysis
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hrv_features = nk.hrv(r_peaks_indices, sampling_rate=self.fs, show=False)
                
                # Add NeuroKit features with 'HRV_' prefix
                for col in hrv_features.columns:
                    if not col.startswith('HRV_'):
                        hrv_dict[f'HRV_{col}'] = hrv_features[col].iloc[0]
                    else:
                        hrv_dict[col] = hrv_features[col].iloc[0]
        except Exception as e:
            print(f"Warning: NeuroKit HRV calculation partial failure: {e}")

        return pd.Series(hrv_dict)


    def extract_statistical_spectral_features(self, ecg_signal):
        """
        Extracts statistical and spectral features directly from the ECG signal.

        Args:
            ecg_signal (np.ndarray): The 1-D ECG signal.

        Returns:
            pd.Series: A Series of statistical and spectral features.
        """
        signal = ecg_signal.copy()

        # Statistical Features
        stats = {
            'Stat_Mean': np.mean(signal),
            'Stat_Std': np.std(signal),
            'Stat_Median': np.median(signal),
            'Stat_Max': np.max(signal),
            'Stat_Min': np.min(signal),
            'Stat_Range': np.max(signal) - np.min(signal),
            'Stat_Skewness': skew(signal),
            'Stat_Kurtosis': kurtosis(signal),
            'Stat_RMS': np.sqrt(np.mean(signal ** 2)),
            'Stat_MAD': np.mean(np.abs(signal - np.mean(signal)))
        }

        # Spectral Features using scipy.signal.welch
        try:
            f, Pxx = welch(signal, fs=self.fs, nperseg=min(256, len(signal)))
            
            # Define frequency bands
            bands = {
                'VLF': (0, 0.05),
                'LF': (0.05, 0.5),
                'MF': (0.5, 5),
                'HF': (5, 40)
            }

            # Calculate power in each band
            for band_name, (low, high) in bands.items():
                band_mask = (f >= low) & (f <= high)
                if np.any(band_mask):
                    stats[f'Spect_Power_{band_name}'] = np.trapezoid(Pxx[band_mask], f[band_mask])
            
            # Total power
            stats['Spect_Total_Power'] = np.trapezoid(Pxx, f)
            
            # Dominant frequency
            stats['Spect_Dominant_Freq'] = f[np.argmax(Pxx)]
            
            # Spectral entropy
            pxx_norm = Pxx / np.sum(Pxx)
            pxx_norm = pxx_norm[pxx_norm > 0]
            stats['Spect_Entropy'] = -np.sum(pxx_norm * np.log2(pxx_norm))

        except Exception as e:
            print(f"Error during spectral calculation: {e}")

        return pd.Series(stats)

    def extract_complexity_features(self, ecg_signal):
        """
        Extracts non-linear complexity features from the signal.

        Args:
            ecg_signal (np.ndarray): The 1-D ECG signal.

        Returns:
            pd.Series: A Series of complexity features.
        """
        rr_data = self._calculate_rr_data(ecg_signal)
        rr_intervals = rr_data['rr_intervals']
        complexity_features = {}

        # Complexity from RR intervals
        if len(rr_intervals) > 20:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Sample Entropy of RR intervals
                    rr_sampen = nk.entropy_sample(rr_intervals, dimension=2, tolerance=0.2 * np.std(rr_intervals))
                    complexity_features['Complexity_RR_SampEn'] = rr_sampen
            except Exception as e:
                print(f"Warning: RR complexity calculation failed: {e}")

        # Complexity from raw signal
        if len(ecg_signal) > 100:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sig_sampen = nk.entropy_sample(ecg_signal, dimension=2, tolerance=0.2 * np.std(ecg_signal))
                    complexity_features['Complexity_Signal_SampEn'] = sig_sampen
            except Exception as e:
                print(f"Warning: Signal complexity calculation failed: {e}")

        return pd.Series(complexity_features)

    def extract_all_features(self, ecg_signal):
        """
        A convenience method to run all feature extraction methods and combine results.
        
        Args:
            ecg_signal (np.ndarray): The 1-D ECG signal.

        Returns:
            pd.Series: A single Series containing all extracted features.
        """
        self._rr_data_cache = {}

        # Extract all feature categories
        basic_f = self.extract_basic_features(ecg_signal)
        hrv_f = self.extract_hrv_features(ecg_signal)
        morph_f = self.extract_morphological_features(ecg_signal)
        stat_spect_f = self.extract_statistical_spectral_features(ecg_signal)
        comp_f = self.extract_complexity_features(ecg_signal)

        # Concatenate all features
        all_features = pd.concat([basic_f, hrv_f, morph_f, stat_spect_f, comp_f])

        return all_features


# Example Usage
if __name__ == '__main__':
    fs = 250
    
    # Load ECG signal
    ecg = np.load('1.1.0/EchoNext_train_waveforms.npy')[0, 0, :, 1]  # Lead II
    
    print(f"ECG signal loaded: {ecg.shape}")
    
    extractor = FeatureExtraction(sampling_rate=fs)
    
    # Extract all features
    full_features = extractor.extract_all_features(ecg)

    # Save to CSV
    df = pd.DataFrame(full_features).transpose()
    df.to_csv('extracted_ecg_features.csv', index=False)
    
    print("\n--- Extracted Features Summary ---")
    print(f"Total features extracted: {len(full_features)}")
    print(f"\nFirst 15 features:")
    print(full_features.head(15).to_string())
    print("\n...")
    print(f"\nLast 10 features:")
    print(full_features.tail(10).to_string())