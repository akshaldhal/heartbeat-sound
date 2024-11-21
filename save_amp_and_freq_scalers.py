import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the processed training dataset
processed_dataset = pd.read_csv("extracted/archive/DHD/final_preprocessed_small.csv")

# Convert the normalized frequencies back to numpy arrays
normalized_freq = np.array([np.array(seq) for seq in processed_dataset["normalized_freq"].apply(eval)])

# Separate frequency and amplitude data
freq_data = normalized_freq[:, :, 0]
amp_data = normalized_freq[:, :, 1]

# Create and fit the scalers
freq_scaler = StandardScaler().fit(freq_data)
amp_scaler = StandardScaler().fit(amp_data)

# Save the scalers
with open("freq_scaler.pkl", "wb") as f:
    pickle.dump(freq_scaler, f)

with open("amp_scaler.pkl", "wb") as f:
    pickle.dump(amp_scaler, f)

print("Scalers saved as 'freq_scaler.pkl' and 'amp_scaler.pkl'")
