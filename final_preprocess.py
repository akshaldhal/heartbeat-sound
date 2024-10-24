import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

j = 0

DROPPING_ARTIFACTS = True

def freq_amp_pair_string_to_list(string):
    """Convert string representation of frequency-amplitude pairs to list."""
    string = string[1:-1]
    string = string.split(", ")
    return [[float(string[i][1:]), float(string[i+1][:-1])] for i in range(0, len(string), 2)]

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

def process_sequences(df, pad_mode='both'):
    """
    Process all sequences in the dataset.
    
    Args:
        df: DataFrame containing sequences
        pad_mode: Padding mode to use
    
    Returns:
        Normalized sequences and the scaler objects used
    """
    # Sort by sequence length
    df = df.copy()
    df["freq_len"] = df["extracted_freq"].apply(len)
    df = df.sort_values(by="freq_len", ascending=False).reset_index(drop=True)
    
    max_seq = df["extracted_freq"].iloc[0]
    max_len = len(max_seq)
    
    # Align and pad sequences
    df["processed_freq"] = df["extracted_freq"].apply(
        lambda x: align_and_pad_sequence(x, max_seq, pad_mode)
    )
    
    # Convert to numpy array
    sequences = np.array([seq for seq in df["processed_freq"]])
    
    # Create scalers for frequencies and amplitudes
    freq_scaler = StandardScaler()
    amp_scaler = StandardScaler()
    
    # Normalize frequencies and amplitudes separately
    normalized = np.zeros_like(sequences)
    normalized[:, :, 0] = freq_scaler.fit_transform(sequences[:, :, 0])  # frequencies
    normalized[:, :, 1] = amp_scaler.fit_transform(sequences[:, :, 1])   # amplitudes
    
    return normalized, freq_scaler, amp_scaler, df

def main():
    # Load the dataset
    processed_dataset = pd.read_csv("extracted/archive/DHD/processed_dataset_small.csv")
    
    # Drop artifacts if specified
    if DROPPING_ARTIFACTS:
        processed_dataset = processed_dataset[processed_dataset["label_value"] != 3]
    
    # Convert string representations to lists
    processed_dataset["extracted_freq"] = processed_dataset["extracted_freq"].apply(freq_amp_pair_string_to_list)
    
    # Process sequences
    normalized_data, freq_scaler, amp_scaler, processed_df = process_sequences(processed_dataset, pad_mode='both')
    
    # Convert normalized sequences back to lists for storing in DataFrame
    processed_df["normalized_freq"] = [seq.tolist() for seq in normalized_data]
    
    # Add flattened version of the normalized sequences
    flattened_data = normalized_data.reshape(normalized_data.shape[0], -1)
    processed_df["flattened_freq"] = [seq.tolist() for seq in flattened_data]
    
    # Save the processed dataset
    output_df = processed_df[[
        'label_value',  # Original label
        'processed_freq',  # Aligned and padded sequences
        'normalized_freq',  # Normalized sequences
        'flattened_freq'  # Flattened normalized sequences
    ]]
    
    # Save the processed dataset
    output_df.to_csv("extracted/archive/DHD/final_preprocessed_small.csv", index=False)
    print("Processing complete. Data saved to 'final_preprocessed_small.csv'")

if __name__ == "__main__":
    main()