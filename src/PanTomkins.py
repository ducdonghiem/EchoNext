import numpy as np
import neurokit2 as nk # The primary tool
import pandas as pd

# Assume 'data_waveforms' is your N x 1 x 2500 x 12 NumPy array
# read from 1.1.0/EchoNext_train_waveforms.npy
data_waveforms = np.load('1.1.0/EchoNext_train_waveforms.npy')
# Assume 'N' is the number of patients
N = data_waveforms.shape[0]
print(N)

FS = 250 # Sampling frequency (250 Hz)
N_LEADS = 12

# Initialize a dictionary to hold all extracted features
all_features_data = {}

for patient_idx in range(N):
    patient_features = {}
    
    # 1. HRV Features (Usually done on the most reliable lead, e.g., Lead II)
    # Lead II is often the 1st index (V1=5, V2=6, ..., I=0)
    lead_II_signal = data_waveforms[patient_idx, 0, :, 1]
    
    # R-Peak detection on Lead II (for reliable HRV)
    _, rpeaks = nk.ecg_peaks(lead_II_signal, sampling_rate=FS)
    
    # Compute and store all HRV metrics
    hrv_features = nk.hrv(rpeaks, sampling_rate=FS)
    patient_features.update(hrv_features.iloc[0].to_dict())

    # 2. Morphological and Statistical Features (Per-Lead)
    lead_features_list = []
    
    for lead_idx in range(N_LEADS):
        current_lead_signal = data_waveforms[patient_idx, 0, :, lead_idx]
        
        # --- R-Peak and Delineation ---
        _, rpeaks_lead = nk.ecg_peaks(current_lead_signal, sampling_rate=FS)
        
        # If no R-peaks found, skip or flag lead as poor quality
        if len(rpeaks_lead['ECG_R_Peaks']) < 2:
            continue
            
        # Get P, Q, S, T indices
        _, waves_lead = nk.ecg_delineate(current_lead_signal, rpeaks_lead, sampling_rate=FS)
        
        # --- Feature Calculation (Morphology) ---
        
        # Calculate QRS duration, PR interval, P-wave amplitude, etc., using 'waves_lead' indices
        morphology_features = calculate_morphology(current_lead_signal, waves_lead) 
        
        # Calculate statistical features (Mean, Std Dev, Skewness, Kurtosis, FFT)
        statistical_features = calculate_stats_and_transforms(current_lead_signal)
        
        # Combine per-lead features
        lead_data = {**morphology_features, **statistical_features}
        lead_data = {f"lead_{lead_idx}_{k}": v for k, v in lead_data.items()} # Prefix features
        lead_features_list.append(lead_data)
        
    # 3. Aggregation and Final Assembly
    # (Steps you will implement next: Mean/Max/Min across all leads for each feature)
    aggregated_features = aggregate_features(lead_features_list)
    patient_features.update(aggregated_features)
    
    # Store the final feature vector
    all_features_data[patient_idx] = patient_features

# Convert to DataFrame for ML/Feature Selection
final_df = pd.DataFrame.from_dict(all_features_data, orient='index')