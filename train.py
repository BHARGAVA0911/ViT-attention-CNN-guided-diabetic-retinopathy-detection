from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
import pandas as pd
from tqdm import tqdm
from torchvision import transforms # Import transforms

#  Constants
NUM_CLASSES = 5
NUM_EPOCHS = 25
BATCH_SIZE = 32
FOLDS = 5
PATIENCE = 5

#  Load Data
csv_path = ""
image_dir = ""
attn_map_dir = ""
df = pd.read_csv(csv_path)

#  Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

#  Track overall metrics
fold_accuracies = []
fold_losses = []

#  Stratified K-Fold Setup
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['diagnosis'])):
    print(f"\nFold {fold+1}/{FOLDS}")

    #  Subset Datasets
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_dataset = OcularDatasetWithAttention(train_df, image_dir, attn_map_dir, transform=transform)
    val_dataset = OcularDatasetWithAttention(val_df, image_dir, attn_map_dir, transform=transform)

    #  Class Weights & Sampler
    class_counts = train_df['diagnosis'].value_counts().sort_index().values
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    sample_weights = train_df['diagnosis'].map(lambda x: class_weights[x]).to_numpy(dtype=np.float32)
    sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    #  Model, Loss, Optimizer
    model = CustomCNNWithAttention(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_val_loss = float("inf")
    best_val_acc = 0
    counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        #  Validation
        model.eval()
        val_loss_total, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_total += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_total / len(val_loader)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1:02d} → Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        #  Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            counter = 0
        elif val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    fold_accuracies.append(best_val_acc)
    fold_losses.append(best_val_loss)
    print(f"Fold {fold+1} Complete → Best Val Acc: {best_val_acc:.4f}, Best Val Loss: {best_val_loss:.4f}")

#  Final Report
print("\n5-Fold Cross-Validation Results:")
print(f"Avg Val Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"Avg Val Loss:     {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}")
