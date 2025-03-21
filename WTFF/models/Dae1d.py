from __future__ import print_function
import torch.nn as nn
import torch

inputflag = 0
# ----------------------------------inputsize == 1024
class encoder(nn.Module):
    def __init__(self, in_channel=1, num_classes=10):
        super(encoder, self).__init__()
        self.in_channel = in_channel
        # Encoder
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))

        self.fc3 = nn.Sequential(
            nn.Linear(1024, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.fc4 = nn.Linear(64, 16)
        self.relu = nn.ReLU()

    def forward(self, x):
        global inputflag
        if x.shape[2] == 1024:
            noise = torch.rand(x.shape, device=x.device) * x.mean() / 10
            x = x + noise
            out = x.view(x.size(0), -1)
            inputflag = 0
            out = self.fc1(out)
        else:
            inputflag = 1
            x = x + torch.rand(x.shape, device=x.device) * x.mean() / 10
            out = x.view(x.size(0), -1)

        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out

class decoder(nn.Module):
    def __init__(self, in_channel=1, num_classes=10):
        super(decoder, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(16, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True))

        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(256, 2048)
        self.fc4 = nn.Linear(2048, 1024)

    def forward(self, z):
        out = self.fc1(z)
        out = self.fc2(out)
        if inputflag ==1:
            out = self.fc3(out)
        else:
            out = self.relu(self.fc3(out))
            out = self.fc4(out)
        return out.view(out.size(0), 1, -1)


class classifier(nn.Module):
    def __init__(self, in_channel=1, num_classes=10):
        super(classifier, self).__init__()
        self.fc = nn.Sequential(nn.ReLU(),  nn.Linear(16, num_classes))

    def forward(self, z):
        label = self.fc(z)
        return label