import torch
import torch.nn as nn
import torch.optim as optim
from autoencoder import AutoEncoder
from modules.losses import SpectralLoss
from dataset import AudioDataset
from torch.utils.data import DataLoader
import glob

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the AutoEncoder model
model = AutoEncoder().to(device)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = SpectralLoss()

# Define your training loop
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for inputs, _ in dataloader:
        inputs = inputs.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        audio, noise, reverbed = model(inputs)

        # Compute the loss
        loss = criterion(reverbed, inputs)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update the running loss
        running_loss += loss.item()

    return running_loss / len(dataloader)

# Example usage
data_list = glob.glob("pitch_encoder/*.wav")
dataset = AudioDataset(data_list, sr=22050, duration=5)
train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # Your training dataloader
num_epochs = 100

for epoch in range(num_epochs):
    loss = train(model, train_dataloader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")