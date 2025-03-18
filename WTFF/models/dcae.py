from __future__ import print_function
from turtle import forward
import torch.nn as nn

class encoder(nn.Module):
    def __init__(self, num_classes=10):
        super(encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=133, stride=5, padding=0),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2, stride=2, return_indices=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2, stride=2, return_indices=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2, stride=2, return_indices=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1),
            nn.MaxPool1d(2, stride=2, return_indices=True),
            nn.ReLU(inplace=True))

    def forward(self, x):
        pool_indices = []
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool1d):
                x, indices = layer.forward(x)
                pool_indices.append(indices)
            else:
                x = layer.forward(x)
        return x, pool_indices


class decoder(nn.Module):
    def __init__(self, num_classes=10):
        super(decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.MaxUnpool1d(2, stride=2),
            nn.ConvTranspose1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxUnpool1d(2, stride=2),
            nn.ConvTranspose1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxUnpool1d(2, stride=2),
            nn.ConvTranspose1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxUnpool1d(2, stride=2),
            nn.ConvTranspose1d(32, 1, kernel_size=133, stride=5, padding=0),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True))

    def forward(self, x, pool_indices):
        for layer in self.decoder.children():
            if isinstance(layer, nn.MaxUnpool1d):
                x = layer.forward(x, indices=pool_indices.pop())
            else:
                x = layer.forward(x)
        return x


class classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(classifier, self).__init__()
        # self.fc1 = nn.Sequential(nn.ReLU(), nn.Linear(24, 256))
        # self.fc2 = nn.Sequential(nn.ReLU(), nn.Linear(256, 64))
        self.fc3 = nn.Sequential(nn.ReLU(), nn.Linear(24*32, 64))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, z, mode=False):
        # z = self.fc1(z)
        # z = self.fc2(z)
        z = self.fc3(z)
        z = self.fc(z)
        return z
