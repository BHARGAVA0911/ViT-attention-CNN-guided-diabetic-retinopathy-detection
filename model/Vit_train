import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import timm  # For Vision Transformer
import os
import cv2
import pandas as pd
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np

# CONFIGURATION

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 20
lr = 1e-4

# LOAD PRETRAINED VIT

model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

# TRAINING LOOP

train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for epoch in range(num_epochs):

    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total * 100
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation Phase
    model.eval()
    val_running_loss, val_correct, val_total =0.0, 0, 0

    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)
            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_labels)

            val_running_loss += val_loss.item()
            _, val_predicted = torch.max(val_outputs, 1)
            val_correct += (val_predicted == val_labels).sum().item()
            val_total += val_labels.size(0)

    val_epoch_loss = val_running_loss / len(val_loader)
    val_acc = val_correct / val_total * 100

    val_losses.append(val_epoch_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs} - "
      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
      f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_acc:.2f}%")
