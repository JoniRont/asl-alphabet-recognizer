import torch
import torch.nn as nn

class HandLandmarkModel(nn.Module):
    def __init__(self, num_classes):
        super(HandLandmarkModel, self).__init__()
        # 21 niveltä * 3 koordinaattia (x, y, z) = 63 syötettä
        self.network = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)