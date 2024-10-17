import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DROPPING_ARTIFACTS = True

def freq_amp_pair_string_to_list(string):
    string = string[1:-1]
    string = string.split(", ")
    return [[float(string[i][1:]), float(string[i+1][:-1])] for i in range(0, len(string), 2)]

def post_pad_list(lst, length):
    return lst + [[0, 0]] * (length - len(lst))

def convert_non_zero_to_one(x):
    if x == 0:
        return 0
    else:
        return 1

processed_dataset = pd.read_csv("extracted/archive/DHD/processed_dataset_small.csv")
if DROPPING_ARTIFACTS:
    processed_dataset = processed_dataset[processed_dataset["label_value"] != 3]

processed_dataset["extracted_freq"] = processed_dataset["extracted_freq"].apply(freq_amp_pair_string_to_list)
processed_dataset["freq_len"] = processed_dataset["extracted_freq"].apply(len)
processed_dataset = processed_dataset.sort_values(by="freq_len", ascending=False).reset_index(drop=False)

max_freq_len = processed_dataset["freq_len"].max()
processed_dataset["extracted_freq"] = processed_dataset["extracted_freq"].apply(post_pad_list, length=max_freq_len)

# Convert the label values to 0 and 1
processed_dataset["label_value"] = processed_dataset["label_value"].apply(convert_non_zero_to_one)

# Normalize the data
scaler = StandardScaler()
normalized_data = np.array([scaler.fit_transform(seq) for seq in processed_dataset["extracted_freq"]])

# Flatten the frequency-amplitude pairs into a single array
flattened_data = normalized_data.reshape(normalized_data.shape[0], -1)

# Split the data
train_data, test_data, train_labels, test_labels = train_test_split(
    flattened_data, 
    processed_dataset["label_value"], 
    test_size=0.2, 
    random_state=42
)

# Custom Dataset class
class HeartSoundDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels.values, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Define the Fully Connected Neural Network (FCNN)
class FCNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(FCNNClassifier, self).__init__()
        layers = []
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.Dropout(0.5))
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            layers.append(nn.Dropout(0.5))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Hyperparameters
input_size = flattened_data.shape[1]  # Length of the flattened input vector
# hidden_sizes = [128, 64]  # full, at 128 batch size
hidden_sizes = [128, 64]  # List of hidden layer sizes
num_classes = len(processed_dataset["label_value"].unique())
batch_size = 128
num_epochs = 20
learning_rate = 0.001

# Prepare dataset and dataloader
train_dataset = HeartSoundDataset(train_data, train_labels)
test_dataset = HeartSoundDataset(test_data, test_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model = FCNNClassifier(input_size, hidden_sizes, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.3)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)
        
        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (torch.round(predicted) == labels).sum().item()
        # print(torch.round(predicted), labels)

    train_accuracy = 100 * correct / total
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (torch.round(predicted) == labels).sum().item()
            # print(torch.round(predicted), labels)
    
    val_accuracy = 100 * correct / total
    scheduler.step(val_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss/len(test_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%')

print("Training complete.")

# Final test
model.eval()
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)
        
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (torch.round(predicted) == labels).sum().item()
        # print(torch.round(predicted), labels)

test_accuracy = 100 * correct / total
print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%')
