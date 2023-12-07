import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, input_length: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_length, input_length),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)


class Discriminator(nn.Module):
    def __init__(self, input_length: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_length, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

