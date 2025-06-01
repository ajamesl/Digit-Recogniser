import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Pooling reduces spatial dims
        self.pool  = nn.MaxPool2d(2, 2)
        # Fully connected layers
        self.fc1 = nn.Linear(64*14*14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x: [batch,1,28,28]
        x = F.relu(self.conv1(x))        # → [b,32,28,28]
        x = self.pool(F.relu(self.conv2(x)))  # → [b,64,14,14]
        x = x.view(x.size(0), -1)        # flatten
        x = F.relu(self.fc1(x))          # hidden
        x = self.fc2(x)                  # logits
        return x
