import torch.nn as nn
import torch.nn.functional as F

class AgeGroupClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)
