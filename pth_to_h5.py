import torch
import h5py
import numpy as np

def convert_pth_to_h5(pth_model_path, h5_model_path):
    # Load the PyTorch model's state dictionary (parameters only)
    state_dict = torch.load(pth_model_path, map_location=torch.device('cpu'))
    
    # Check if it's a full model or just a state_dict
    if hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()
    
    # Create a new HDF5 file
    with h5py.File(h5_model_path, 'w') as h5_file:
        for name, param in state_dict.items():
            # Convert the parameter to a numpy array and save it in the HDF5 file
            h5_file.create_dataset(name, data=param.cpu().numpy())

model_name = "trained_fcnn_classifier_model_128_to_16_7281"

convert_pth_to_h5(model_name + '.pth', model_name + '.h5')
