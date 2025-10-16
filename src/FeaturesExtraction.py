import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy.stats import skew, kurtosis

# The standard sampling rate for the EchoNext dataset is 250 Hz (10s segment).
# We will use this as the default if not provided.
DEFAULT_FS = 250

class FeatureExtraction:
    """
    A class for extracting various categories of features from a single-lead ECG signal.
    
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
        # A cache for R-peak and RR interval data to avoid re-calculating the foundation
        # when calling multiple feature methods.
        self._rr_data_cache = {}

    def _calculate_rr_data(self, ecg_signal):
        """
        Internal utility function (Category 0) to detect R-peaks and compute RR intervals.
        Caches the result to prevent redundant computation across feature methods.

        Args:
            ecg_signal (np.ndarray): The 1-D ECG signal.

        Returns:
            dict: Contains 'rr_intervals' (list of ms) and 'r_peaks' (list of indices).
        """

        # Check cache first
        if len(self._rr_data_cache) > 0:
            return self._rr_data_cache

        # Convert NumPy array to NeuroKit's expected format (Pandas Series)
        signal = pd.Series(ecg_signal)

        try:
            # Step 1: Process the signal (cleaning is optional but recommended)
            cleaned_ecg = nk.ecg_clean(signal, sampling_rate=self.fs)
            
            # Step 2: Detect R-peaks (using the default Pan-Tompkins method)
            # R-peaks are returned as a dictionary of indices.
            _, r_peaks = nk.ecg_peaks(cleaned_ecg, sampling_rate=self.fs)
            r_peaks_indices = r_peaks['ECG_R_Peaks']

            # Step 3: Compute heart rate and RR intervals
            # rri (RR intervals) are in milliseconds (ms)
            # rr_intervals = nk.hrv_get_rri(r_peaks_indices, sampling_rate=self.fs, show=False)
            rate = nk.ecg_rate(r_peaks_indices, sampling_rate=self.fs, show=False)
            # Convert to RR intervals in milliseconds
            rr_intervals = (np.diff(r_peaks_indices) / self.fs) * 1000

            # Store in cache
            self._rr_data_cache = {
                'rr_intervals': rr_intervals,
                'r_peaks': r_peaks_indices,
                'cleaned_ecg': cleaned_ecg
            }
            return self._rr_data_cache

        except Exception as e:
            print(f"Error during R-peak detection or RR interval calculation: {e}")
            # Return empty data structures if calculation fails
            return {'rr_intervals': [], 'r_peaks': np.array([]), 'cleaned_ecg': signal}


    def extract_hrv_features(self, ecg_signal):
        """
        Extracts Heart Rate Variability (HRV) features (Category 1).
        Requires RR intervals, which are calculated internally.

        Args:
            ecg_signal (np.ndarray): The 1-D ECG signal.

        Returns:
            pd.Series: A Series of HRV features (Time, Frequency, Non-Linear).
        """
        rr_data = self._calculate_rr_data(ecg_signal)
        r_peaks_indices = rr_data['r_peaks']

        if len(r_peaks_indices) < 5:
            print("Warning: Too few R-peaks detected for reliable HRV calculation.")
            return pd.Series({}) # Return empty series

        try:
            # NeuroKit2 function to extract all standard HRV features
            hrv_features = nk.hrv(r_peaks_indices, sampling_rate=self.fs)
            
            # Since hrv returns a DataFrame with one row, we extract the Series.
            return hrv_features.iloc[0].rename(lambda x: f"HRV_{x}")

        except Exception as e:
            print(f"Error during HRV calculation: {e}")
            return pd.Series({})


    def extract_morphological_features(self, ecg_signal):
        """
        Extracts morphological features (Category 2) such as P, QRS, and T characteristics.
        This relies on R-peak detection and full wave delineation.

        Args:
            ecg_signal (np.ndarray): The 1-D ECG signal.

        Returns:
            pd.Series: A Series of morphological features (e.g., durations, amplitudes).
        """
        rr_data = self._calculate_rr_data(ecg_signal)
        r_peaks_indices = rr_data['r_peaks']
        cleaned_ecg = rr_data['cleaned_ecg']

        if len(r_peaks_indices) == 0:
             print("Warning: No R-peaks detected for morphological feature calculation.")
             return pd.Series({})

        try:
            # Delineate waves (P, Q, S, T onset and offset)
            _, waves = nk.ecg_delineate(cleaned_ecg, r_peaks_indices, sampling_rate=self.fs, show=False, method="cwt")
            
            # Analyze features based on waves
            features = nk.ecg_analyze(cleaned_ecg, r_peaks_indices, sampling_rate=self.fs, waves=waves)
            
            # The result is a DataFrame with many complex features, including the one 
            # mentioned in your tabular data (QRS duration, PR interval, QT corrected).
            
            # Filter for key morphological features (Averages across beats)
            # The 'ECG_P_Onsets' etc. are lists per beat; the 'ECG_P_Wave_Duration_Mean' are the aggregates.
            
            # Note: We assume the desired output includes the *average* feature values.
            # We must ensure the columns exist before accessing them
            
            morph_features = {}
            key_prefixes = ['Duration', 'Amplitude', 'Interval', 'Rate'] 
            
            # Select relevant mean features calculated by nk.ecg_analyze
            relevant_cols = [col for col in features.columns 
                             if features[col].dtype != 'object' and any(prefix in col for prefix in key_prefixes)]
                             
            # Take the mean of these columns across all detected beats
            for col in relevant_cols:
                morph_features[f'Morph_{col}'] = features[col].mean()

            # Ensure we get the core features mentioned in the tabular data:
            if 'ECG_PR_Interval_Mean' in features.columns:
                morph_features['Morph_PR_Interval_Mean'] = features['ECG_PR_Interval_Mean'].mean()
            if 'ECG_QRS_Duration_Mean' in features.columns:
                morph_features['Morph_QRS_Duration_Mean'] = features['ECG_QRS_Duration_Mean'].mean()
            if 'ECG_QT_Corrected_Mean' in features.columns:
                morph_features['Morph_QT_Corrected_Mean'] = features['ECG_QT_Corrected_Mean'].mean()
            if 'ECG_Heart_Rate_Mean' in features.columns:
                 morph_features['Morph_Heart_Rate_Mean'] = features['ECG_Heart_Rate_Mean'].mean()

            return pd.Series(morph_features)

        except Exception as e:
            print(f"Error during morphological feature calculation: {e}")
            return pd.Series({})


    def extract_statistical_spectral_features(self, ecg_signal):
        """
        Extracts statistical (Time Domain) and spectral (Frequency Domain) features 
        directly from the ECG signal (Category 3).

        Args:
            ecg_signal (np.ndarray): The 1-D ECG signal.

        Returns:
            pd.Series: A Series of statistical and spectral features.
        """
        signal = ecg_signal.copy() # Work on a copy

        # --- 3A: Statistical Features (Signal Distribution) ---
        stats = {
            'Stat_Mean': np.mean(signal),
            'Stat_Std': np.std(signal),
            'Stat_Median': np.median(signal),
            'Stat_Max': np.max(signal),
            'Stat_Min': np.min(signal),
            'Stat_Range': np.max(signal) - np.min(signal),
            'Stat_Skewness': skew(signal),
            'Stat_Kurtosis': kurtosis(signal)
        }

        # --- 3B: Spectral Features (Frequency Content) ---
        # Using Welch's method for Power Spectral Density (PSD)
        try:
            # Calculate PSD
            f, Pxx = nk.signal_psd(signal, sampling_rate=self.fs, method="welch", show=False)
            
            # Define standard frequency bands (e.g., low, medium, high frequency content)
            # These ranges are illustrative and may need tuning based on specific research.
            bands = {
                'VLF': (0, 0.05), # Very Low Frequency (Related to long-term regulatory mechanisms)
                'LF': (0.05, 0.5), # Low Frequency (Often related to muscle tremor/noise, but can be informative)
                'HF': (0.5, 40)   # High Frequency (Includes primary ECG frequencies)
            }

            # Calculate power in each band
            for band_name, (low, high) in bands.items():
                band_mask = (f >= low) & (f <= high)
                # Area under the curve (integrated power)
                stats[f'Spect_Power_{band_name}'] = np.trapz(Pxx[band_mask], f[band_mask])

        except Exception as e:
            print(f"Error during Spectral (PSD) calculation: {e}")
            # Skip spectral features if error occurs

        return pd.Series(stats)


    def extract_complexity_features(self, ecg_signal):
        """
        Extracts non-linear complexity features (Category 4) from the signal or RR intervals.

        Args:
            ecg_signal (np.ndarray): The 1-D ECG signal.

        Returns:
            pd.Series: A Series of complexity features (e.g., Sample Entropy).
        """
        rr_data = self._calculate_rr_data(ecg_signal)
        r_peaks_indices = rr_data['r_peaks']
        complexity_features = {}

        # --- 4A: Complexity from RR intervals (Rhythm Complexity) ---
        if len(r_peaks_indices) > 20: # Needs sufficient data points
            try:
                # Approximate Entropy of RR intervals
                rr_entropy = nk.entropy(rr_data['rr_intervals'], dimension=2, r=0.2 * np.std(rr_data['rr_intervals']),
                                        method="ApEn")
                complexity_features['Complexity_RR_ApEn'] = rr_entropy.iloc[0]['ApEn']
            except Exception as e:
                print(f"Error calculating RR Complexity features: {e}")

        # --- 4B: Complexity from Raw Signal ---
        try:
            # Sample Entropy of the raw signal
            sig_entropy = nk.entropy(ecg_signal, dimension=2, r=0.2 * np.std(ecg_signal), method="SampEn")
            complexity_features['Complexity_Signal_SampEn'] = sig_entropy.iloc[0]['SampEn']
        except Exception as e:
            print(f"Error calculating Signal Complexity features: {e}")

        return pd.Series(complexity_features)

    def extract_all_features(self, ecg_signal):
        """
        A convenience method to run all feature extraction methods and combine results.
        
        Args:
            ecg_signal (np.ndarray): The 1-D ECG signal.

        Returns:
            pd.Series: A single Series containing all extracted features.
        """
        # Clear cache before starting a new extraction
        self._rr_data_cache = {}

        # 1. HRV
        hrv_f = self.extract_hrv_features(ecg_signal)
        
        # 2. Morphological
        morph_f = self.extract_morphological_features(ecg_signal)
        
        # 3. Statistical/Spectral
        stat_spect_f = self.extract_statistical_spectral_features(ecg_signal)
        
        # 4. Complexity
        # comp_f = self.extract_complexity_features(ecg_signal)

        # Concatenate all features into a single Series
        all_features = pd.concat([hrv_f, morph_f, stat_spect_f])
        
        # Add the mean heart rate explicitly as a basic feature (if available)
        if 'Morph_Heart_Rate_Mean' in all_features.index:
             all_features['Basic_Heart_Rate_BPM'] = all_features['Morph_Heart_Rate_Mean']

        return all_features

# Example Usage (Demonstration - requires a mock ECG signal)
if __name__ == '__main__':
    # Mock 10-second ECG signal at 250 Hz (2500 samples)
    fs = 250
    # duration = 10
    # samples = fs * duration
    
    # Generate a simulated ECG signal
    # ecg = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=80)

    ecg = np.load('1.1.0/EchoNext_train_waveforms.npy')[0, 0, :, 1]  # Lead II of the first patient
    
    print(f"ECG signal created: {ecg.shape}")
    
    extractor = FeatureExtraction(sampling_rate=fs)
    
    # Extract all features
    full_features = extractor.extract_all_features(ecg)

    # save to CSV
    df = pd.DataFrame(full_features).transpose()
    df.to_csv('extracted_ecg_features.csv', index=False)
    
    print("\n--- Extracted Features Summary ---")
    print(f"Total features extracted: {len(full_features)}")
    print(full_features.head(10).to_markdown(numalign="left", stralign="left"))
    print("...")
    print(full_features.tail(5).to_markdown(numalign="left", stralign="left"))
