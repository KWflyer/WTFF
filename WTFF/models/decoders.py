import torch
import torch.nn as nn

class decoder_resnet18(nn.Module):
    def __init__(self, num_classes=10):
        super(decoder_resnet18, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        for layer in self.decoder.children():
            x = layer.forward(x)
        return x

class classifier_resnet18(nn.Module):
    def __init__(self, num_classes=10):
        super(classifier_resnet18, self).__init__()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, z):
        z = self.fc(z)
        return z