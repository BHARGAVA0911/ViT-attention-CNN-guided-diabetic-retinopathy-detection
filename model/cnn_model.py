
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNNWithAttention(nn.Module):
    def __init__(self, num_classes):
        super(CustomCNNWithAttention, self).__init__()

        # 🔹 4-Channel Input (RGB + Attention Map)
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # 🔹 Extra Conv Layer to fix channel mismatch after CReLU
        self.reduction_conv = nn.Conv2d(256, 128, kernel_size=1)  # Reduce 256 → 128

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 8 * 8, 32)  # Adjusted for 256x256 input
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.pool(torch.tanh(self.bn1(self.conv1(x))))
        x = self.pool(F.softsign(self.bn2(self.conv2(x))))
        x = self.pool(F.elu(self.bn3(self.conv3(x))))

        # 🔹 Fix for CReLU: Split into two channels & apply ReLU separately
        x = self.bn4(self.conv4(x))
        x = torch.cat((F.relu(x), F.relu(-x)), dim=1)  # CReLU manually implemented
        x = self.pool(x)

        # 🔹 Reduce channels from 256 → 128
        x = self.reduction_conv(x)

        x = self.pool(F.relu6(self.bn5(self.conv5(x))))  # Use `F.relu6()`

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# 🔹 Model Initialization
num_classes = 5  # Adjust based on dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNNWithAttention(num_classes).to(device)

print("✅ Model Ready")
