import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) # 32x32 -> 32x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # 16x16 -> 16x16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                 # 16x16 -> 8x8
        self.fc1 = nn.Linear(32 * 8 * 8, 128)                              # Flatten -> 128
        self.fc2 = nn.Linear(128, num_classes)                             # 128 -> num_classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)   # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    