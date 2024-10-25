# Needs to be updated to work with the new model format.

import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

DELTA_TIME = 0.01
MAX_FREQ_LEN = 47633
# extracted/archive/DHD/audio/normal_2023_0.wav

def post_pad_list(lst, length):
    return lst + [[0, 0]] * (length - len(lst))

def flatten_2d_list(lst):
  list = []
  for sublist in lst:
    for item in sublist:
      list.append(item)

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


# audio_filename = "example.wav"
audio_filename = "extracted/archive/DHD/audio/abnormal_s3_2023_0.wav"
audio_wave_data = extract_frequencies(audio_filename, DELTA_TIME)
if len(audio_wave_data) < MAX_FREQ_LEN:
  audio_wave_data = post_pad_list(audio_wave_data, MAX_FREQ_LEN)
elif len(audio_wave_data) > MAX_FREQ_LEN:
  audio_wave_data = audio_wave_data[:MAX_FREQ_LEN]

audio_wave_data = flatten_2d_list(audio_wave_data)