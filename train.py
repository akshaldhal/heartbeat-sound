import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import json

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

def main():
    # Load the preprocessed dataset
    processed_dataset = pd.read_csv("extracted/archive/DHD/final_preprocessed_small.csv")
    
    # Convert string representation of flattened data back to numpy array
    flattened_data = np.array([eval(seq) for seq in processed_dataset["flattened_freq"]])
    
    # Split the data
    train_data, test_data, train_labels, test_labels = train_test_split(
        flattened_data, 
        processed_dataset["label_value"], 
        test_size=0.2, 
        random_state=42
    )

    # Hyperparameters
    input_size = flattened_data.shape[1]
    # hidden_sizes = [64, 32, 16] for small set, batch size 32, 71.05% accuracy
    # hidden_sizes = [16, 8] for small set, batch size 64, 73.68% accuracy          WINNER
    
    hidden_sizes = [16, 8]
    num_classes = len(processed_dataset["label_value"].unique())+1
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.001

    # Prepare dataset and dataloader
    train_dataset = HeartSoundDataset(train_data, train_labels)
    test_dataset = HeartSoundDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
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
        
        val_accuracy = 100 * correct / total
        scheduler.step(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss/len(test_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%')

    print("Training complete.")

    # Save the trained model
    model_path = "trained_fcnn_classifier_model_128_to_16_.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

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

    test_accuracy = 100 * correct / total
    print(f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%')

if __name__ == "__main__":
    main()