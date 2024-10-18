"""
Script to train LTAE model on PASTIS
"""

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from src.data.dataset import SITSDataset, MiniBatchSITSDataset  # Assuming you have a dataset class
from src.models.ltae import LTAE  # The model file
from src.training.utils import validate_model  # A utility function to validate the model


model_weights_path = "/mnt/c/repos/am4cm/.checkpoints/best_model_LTAE.pth"
pastis_path = "/mnt/c/data/PASTIS-R/PASTIS-R"
device = "cuda"

def main():
    ### Create datasets and data loaders
    train_dataset = MiniBatchSITSDataset(SITSDataset(pastis_path, [1,2,3], patch_frac=1, pixel_frac=0.2), patches_per_group=32)
    val_dataset = MiniBatchSITSDataset(SITSDataset(pastis_path, 4, patch_frac=0.1, pixel_frac=0.05), patches_per_group=32)
    eval_dataset = MiniBatchSITSDataset(SITSDataset(pastis_path, 5, patch_frac=0.1, pixel_frac=0.05), patches_per_group=32)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=512, shuffle=False, num_workers=4)


    # Initialize the model
    model = LTAE(
        in_channels=10,  # Adjust according to your dataset
        n_head=8,
        d_k=16,
        mlp3=[256, 128],
        mlp4=[128, 64, 32],
        n_classes=20,
        dropout=0.1,
        d_model=256,
        T=1000
    )
    model.to(device)


    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    # Training loop
    num_epochs = 5

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

        # Validation
        val_loss = validate_model(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            print("Saving model...")
            torch.save(model.state_dict(), model_weights_path)
            best_val_loss = val_loss

    # Evaluate the model
    model.load_state_dict(torch.load(model_weights_path))
    validate_model(model, eval_loader, criterion, device, evaluate=True, evaluation_path="../evals/best_model_LTAE_{}")


if __name__ == "__main__":
    main()
