import torch.nn as nn

class AgeRegressor(nn.Module):
    def __init__(self, input_dim):
        super(AgeRegressor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)
