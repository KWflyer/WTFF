from __future__ import print_function
from turtle import forward
import torch.nn as nn

class encoder(nn.Module):
    def __init__(self, num_classes=10):
        super(encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 3, kernel_size=35, stride=1, padding=1),
            nn.BatchNorm1d(3),
            nn.MaxPool1d(2, stride=2, return_indices=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(3, 32, kernel_size=3, stride=1, padding=1),
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
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2, stride=2, return_indices=True),
            nn.ReLU(inplace=True))

    def forward(self, x):
        features = [x]
        pool_indices = []
        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool1d):
                x, indices = layer.forward(x)
                pool_indices.append(indices)
            elif isinstance(layer, nn.ReLU):
                x = layer.forward(x)
                features.append(x)
            else:
                x = layer.forward(x)
        return [features[0], features[2], features[4], features[5]], pool_indices


class decoder(nn.Module):
    def __init__(self, num_classes=10):
        super(decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.MaxUnpool1d(2, stride=2),
            nn.ConvTranspose1d(32, 32, kernel_size=3, stride=1, padding=1),
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
            nn.ConvTranspose1d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.MaxUnpool1d(2, stride=2),
            nn.ConvTranspose1d(3, 1, kernel_size=35, stride=1, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True))

    def forward(self, features, pool_indices):
        i = 0
        for layer in self.decoder.children():
            if isinstance(layer, nn.MaxUnpool1d):
                if i == 0:
                    x = layer.forward(features.pop(), indices=pool_indices.pop())
                elif i in [1, 3]:
                    x = layer.forward(x + features.pop(), indices=pool_indices.pop())
                else:
                    x = layer.forward(x, indices=pool_indices.pop())
                i += 1
            else:
                x = layer.forward(x)
        return x + features[0]


class classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(classifier, self).__init__()
        self.fc1 = nn.Sequential(nn.ReLU(), nn.Linear(32*63, 256))
        self.fc2 = nn.Sequential(nn.ReLU(), nn.Linear(256, 64))
        self.fc3 = nn.Sequential(nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, z):
        z = self.fc1(z)
        z = self.fc2(z)
        z = self.fc3(z)
        return z
