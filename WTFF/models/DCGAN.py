import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm as SN

class Generator(nn.Module):
    def __init__(self, nz=1, ngf=8, nc=1):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            SN(nn.ConvTranspose1d(nz, ngf * 8, 4, 2, 1, bias=False)),
            SN(nn.ConvTranspose1d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)),
            SN(nn.ConvTranspose1d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)),
            SN(nn.ConvTranspose1d(ngf * 2, ngf, 4, 2, 1, bias=False)),
            SN(nn.ConvTranspose1d(ngf, nc, 145, 1, 0, bias=False)),
        )
    
    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, nc=1, ndf=8):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, ndf, 145, 1, 0, bias=False),
            nn.Conv1d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.Conv1d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.Conv1d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.Conv1d(ndf * 8, nc, 4, 2, 1, bias=False),
            nn.Linear(119, 1),
            nn.Sigmoid()
        )
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.model(x)


class Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
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
            AdaptiveLinear(num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class AdaptiveLinear(nn.Module):
    def __init__(self, num_classes=10):
        super(AdaptiveLinear, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(32, num_classes)
    
    def forward(self, x):
        return self.linear(self.avgpool(x).reshape(x.size(0),-1))

class DCGAN():
    def __init__(self, input_dim=1, num_channels=1, num_classes=10):
        super(DCGAN, self).__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.generator = Generator(input_dim, num_channels, num_classes)
        self.discriminator = Discriminator(input_dim, num_channels, num_classes)