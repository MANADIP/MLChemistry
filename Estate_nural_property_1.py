import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
Data = pd.read_csv('dataset_train.csv')
smiles = Data.iloc[:, 0]
mols = [Chem.MolFromSmiles(smile) for smile in smiles]

# Calculate EState fingerprints 
fingerprints = []
for mol in mols:
    fingerprint = FingerprintMol(mol)
    fingerprints.append(fingerprint)

x = fingerprints
print(len(x))
y = Data.iloc[:, 1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

x_train = np.array(x_train)                          # Convert to numpy array
x_train = torch.FloatTensor(x_train)                 # Convert to tensor
x_train = x_train.view(len(x_train), -1)             # Reshape into (num_samples, num_features)

x_test = np.array(x_test)                            # Convert to numpy array
x_test = torch.FloatTensor(x_test)                   # Convert to tensor
x_test = x_test.view(len(x_test), -1)

# Convert target variables to numpy arrays and then  torch
y_train = torch.FloatTensor(np.array(y_train).reshape(-1, 1))
y_test = torch.FloatTensor(np.array(y_test).reshape(-1, 1))

# Model definition
class Model(nn.Module):
    def __init__(self, in_feature=len(x_train[0]), H1=100, out_feature=1):
        super().__init__()
        self.fc1 = nn.Linear(in_feature, H1)
        self.out = nn.Linear(H1, out_feature)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.out(x)
        return x

# Initialize the model
torch.manual_seed(40)
model = Model(in_feature=x_train.shape[1])          # Pass input feature size to the model

# Loss function
criterion = nn.MSELoss()

# Optimizer
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training
epochs = 1000
losses_train = []
losses_test = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    # Forward 
    y_pred_train = model(x_train)
    # Loss calculation
    loss_train = criterion(y_pred_train, y_train)
    # Backward pass
    loss_train.backward()
    optimizer.step()
    # Save train loss
    losses_train.append(loss_train.item())

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_pred_test = model(x_test)
        loss_test = criterion(y_pred_test, y_test)
        # Save test loss
        losses_test.append(loss_test.item())

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Train Loss: {loss_train.item()}, Test Loss: {loss_test.item()}')

# Plot losses
plt.plot(range(epochs), losses_train, label='Train Loss')
plt.plot(range(epochs), losses_test, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot actual vs predicted values
with torch.no_grad():
    model.eval()
    y_pred_train = model(x_train)
    y_pred_test = model(x_test)

plt.figure(figsize=(12, 6))

# Training set
plt.subplot(1, 2, 1)
plt.scatter(y_train.numpy(), y_pred_train.numpy())
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='-')
plt.title('Actual vs Predicted (Training)')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Test set
plt.subplot(1, 2, 2)
plt.scatter(y_test.numpy(), y_pred_test.numpy())
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='-')
plt.title('Actual vs Predicted (Test)')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.tight_layout()
plt.show()
#print(len(x_train))
    # Save train loss
