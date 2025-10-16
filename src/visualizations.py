import numpy as np

import matplotlib.pyplot as plt

# Load the .npy file
file_path = "1.1.0/EchoNext_train_waveforms.npy"
data_waveforms = np.load(file_path)

# Specify the patient index
patient_idx = 0  # Change this index as needed

# Extract the lead_II_signal
lead_II_signal = data_waveforms[patient_idx, 0, :, 1]

# Plot the ECG graph
plt.figure(figsize=(10, 6))
plt.plot(lead_II_signal, label="Lead II Signal")
plt.title("ECG Graph - Lead II Signal")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.show()