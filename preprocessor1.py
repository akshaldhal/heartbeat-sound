import librosa
import numpy as np
import pickle
import pandas as pd

DELTA_TIME = 0.001

def extract_frequencies(filename, delta_time, method='dominant'):
  y, sr = librosa.load(filename, sr=None)
  frame_length = int(delta_time * sr)
  frame_length = 2 ** int(np.floor(np.log2(frame_length)))
  hop_length = frame_length // 4
  stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
  amplitudes = np.abs(stft)
  freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    
  if amplitudes.shape[0] != len(freqs):
    raise ValueError(f"Mismatch between frequency bins and amplitude frames.")
    
  frequency_amplitude_pairs = []
    
  for frame_amplitudes in amplitudes.T:
    # Method to select the representative frequency for this frame
    if method == 'dominant':
      # Get the frequency with the maximum amplitude
      max_amp_index = np.argmax(frame_amplitudes)
      representative_freq = float(freqs[max_amp_index])
      representative_amp = float(frame_amplitudes[max_amp_index])        
    elif method == 'average':
      # Weighted average frequency
      total_amp = np.sum(frame_amplitudes)
      if total_amp == 0:
        representative_freq = 0
        representative_amp = 0
      else:
        representative_freq = float(np.sum(freqs * frame_amplitudes) / total_amp)
        representative_amp = float(np.mean(frame_amplitudes))        
    elif method == 'median':
      # Median frequency (based on sorted amplitudes)
      sorted_indices = np.argsort(frame_amplitudes)
      median_index = sorted_indices[len(sorted_indices) // 2]
      representative_freq = float(freqs[median_index])
      representative_amp = float(np.median(frame_amplitudes))
    else:
      raise ValueError(f"Invalid method: {method}. Choose from 'dominant', 'average', or 'median'.")        
    
    # Append the representative frequency and amplitude as a pair
    frequency_amplitude_pairs.append((representative_freq, representative_amp))
  
  return frequency_amplitude_pairs


audio_path = "extracted/archive/DHD/audio/"

label_map = {"normal" : 0, "extrahls" : 1, "murmur" : 2, "artifact" : 3, "extrastole" : 4}

labels = open("extracted/archive/DHD/labels_edit.csv", "r")

processed_dataset = [["filename", "label", "label_value", "extracted_freq"]]

count = 0
for line in labels:
  entry = line.strip().split(",")
  audio_filename = audio_path + entry[1]
  audio_label = entry[2]
  audio_wave_data = extract_frequencies(audio_filename, DELTA_TIME)
  audio_label_value = label_map[audio_label]
  processed_dataset.append([audio_filename, audio_label, audio_label_value, audio_wave_data])
#   print(audio_wave_data)
  count += 1
  # if (count > 3):
  #   break
  print(f"Processed {count} files.")

df = pd.DataFrame(processed_dataset[1:], columns=processed_dataset[0])
# df.to_hdf("extracted/archive/DHD/processed_dataset.h5", key="df", mode="w")
df.to_csv("extracted/archive/DHD/processed_dataset.csv", index=False)


# with open("extracted/archive/DHD/processed_dataset.pkl", "wb") as f:
#     pickle.dump(processed_dataset, f)

print("Processing done!")