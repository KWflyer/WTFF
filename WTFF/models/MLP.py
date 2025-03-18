from __future__ import print_function
import torch.nn as nn

inputflag = 0
# ----------------------------------inputsize == 1024
class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.dim_change = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))

        self.fc3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))

        self.fc4 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True))

        self.fc5 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))

        self.fc6 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        global inputflag
        if x.shape[2] == 512:
            inputflag = 0
            out = self.dim_change(x)
            out = self.fc1(out)
        else:
            inputflag = 1
            out = self.dim_change(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        out = self.fc(out)
        return out


class MLP_Fusion(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(MLP_Fusion, self).__init__()
        self.in_channel = in_channel
        self.fc1 = nn.Sequential(
            nn.Linear(self.in_channel, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(64, out_channel),)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class MLP_Sim(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(MLP_Sim, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_channel, out_channel),)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        return out

