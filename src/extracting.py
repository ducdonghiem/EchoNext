from FeaturesExtraction3 import FeatureExtraction
import numpy as np
import pandas as pd

def extract_features_train():
    fs = 250
    
    # take a sample of 20 ECGs from the training set
    
    ecg_train = np.load('1.1.0/EchoNext_train_waveforms.npy')[0:10, :, :, :]

    # loop all N samples
    print(ecg_train.shape)

    for i in range(ecg_train.shape[0]):
        # Load ECG signal (Lead II)
        ecg = ecg_train[i, 0, :, 1]
        
        print(f"Processing sample {i+1}/{ecg_train.shape[0]}: ECG signal shape: {ecg.shape}")
        
        extractor = FeatureExtraction(sampling_rate=fs)
        
        # Extract all features
        features = extractor.extract_all_features(ecg)
        
        # Save to CSV
        df = pd.DataFrame(features).transpose()

        # save each sample as a row in a CSV file
        if i == 0:
            # df.to_csv('extracted_ecg_features_train.csv', mode='w', header=True, index=False)
            df.to_csv('test.csv', mode='w', header=True, index=False)
        else:
            # df.to_csv('extracted_ecg_features_train.csv', mode='a', header=False, index=False)
            df.to_csv('test.csv', mode='a', header=False, index=False)
    
    
if __name__ == "__main__":
    extract_features_train()