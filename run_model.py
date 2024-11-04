import librosa
import numpy as np
import pandas as pd

# DELTA_TIME = 0.01
# MAX_FREQ_LEN = 47633
# extracted/archive/DHD/audio/normal_2023_0.wav

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

def align_and_pad_sequence(current_seq, max_len_seq, pad_mode='both'):
    global j
    j += 1
    print(j)
    """
    Aligns a sequence with a reference sequence and adds padding.
    
    Args:
        current_seq: The sequence to align and pad
        max_len_seq: The reference sequence to align against
        pad_mode: 'both', 'front', or 'back' padding
    
    Returns:
        Aligned and padded sequence
    """
    if len(current_seq) == len(max_len_seq):
        return current_seq
    
    # Convert sequences to numpy arrays for easier manipulation
    curr_array = np.array(current_seq)
    max_array = np.array(max_len_seq)
    
    # Find best alignment position
    min_euclidean_distance = float('inf')
    best_start_idx = 0
    
    for i in range(len(max_len_seq) - len(current_seq) + 1):
        window = max_array[i:i+len(current_seq)]
        distance = np.linalg.norm(curr_array - window)
        if distance < min_euclidean_distance:
            min_euclidean_distance = distance
            best_start_idx = i
    
    # Calculate padding sizes
    total_pad = len(max_len_seq) - len(current_seq)
    
    if pad_mode == 'both':
        front_pad = best_start_idx
        back_pad = total_pad - front_pad
    elif pad_mode == 'front':
        front_pad = total_pad
        back_pad = 0
    else:  # pad_mode == 'back'
        front_pad = 0
        back_pad = total_pad
    
    # Create padded sequence
    padded_seq = ([[0, 0]] * front_pad + 
                  current_seq + 
                  [[0, 0]] * back_pad)
    
    return padded_seq


if __name__ == "__main__":
    main()