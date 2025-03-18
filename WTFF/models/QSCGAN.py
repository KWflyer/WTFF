import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm as SN

class Generator(nn.Module):
    def __init__(self,nz, ngf, nc):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            SN(nn.ConvTranspose1d(nz, ngf * 8, 4, 2, 1, bias=False)),
            SN(nn.ConvTranspose1d(ngf * 8, ngf * 2, 4, 2, 1, bias=False)),
            SN(nn.ConvTranspose1d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)),
            SN(nn.ConvTranspose1d(ngf * 2, ngf * 2, 4, 2, 1, bias=False)),
            SN(nn.ConvTranspose1d(ngf, nc, 144, 1, 0, bias=False)),
        )
    
    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        # 输入的信号长度：1824
        self.model = nn.Sequential(
            nn.Conv1d(1, ndf, 144, 1, 0, bias=False),
            nn.Conv1d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.Conv1d(ndf * 8, 1, 4, 2, 1, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.model(x)


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


class Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        # 输入的信号长度：1824
        self.model = nn.Sequential(
            nn.Conv1d(1, 128, 5, 1, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 128, 5, 1, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(128, 128, 5, 1, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(128, 64, 5, 1, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 64, 5, 1, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(64, 64, 5, 1, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 32, 5, 1, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 32, 5, 1, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 32, 5, 1, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2, 2),
            nn.AdaptiveAvgPool1d(1),
            Flatten(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)
