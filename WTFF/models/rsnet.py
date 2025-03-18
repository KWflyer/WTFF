from __future__ import absolute_import

import torch.nn as nn
import torch

class CWBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=4):
        super(CWBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(planes, planes * self.expansion, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(planes)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(planes, planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(planes // reduction, planes, bias=False),
            nn.Sigmoid()
        )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        absout = torch.abs(out)
        gapout = self.avg_pool(absout)

        b, c, _ = gapout.size()
        gapout = gapout.view(b, c)
        scales = self.fc(gapout)

        thres = gapout * scales
        thres = thres.view(b, c, 1)

        sub = absout - thres
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        out = out * n_sub

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class CSBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=4):
        super(CSBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(planes, planes * self.expansion, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(planes)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(planes, planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(planes // reduction, 1, bias=False),
            nn.Sigmoid()
        )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        absout = torch.abs(out)
        gapout = self.avg_pool(absout)

        b, c, _ = gapout.size()
        gapout = gapout.view(b, c)
        scales = self.fc(gapout)

        thres = scales * torch.mean(gapout, dim=1).view(b, 1)

        sub = absout - thres.view(b, 1, 1)
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        out = out * n_sub

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class RSNet(nn.Module):

    def __init__(self, num_classes=10, block=CWBlock):
        super(RSNet, self).__init__()
        self.inplanes = 4

        self.conv1 = nn.Conv1d(1, 4, kernel_size=3, stride=2, bias=False, padding=1)
        self.bn1 = nn.BatchNorm1d(4)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 4, 3, stride=2)
        self.layer2 = self._make_layer(block, 8, 3, stride=2)
        self.layer3 = self._make_layer(block, 16, 3, stride=2)

        # self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

if __name__ == '__main__':
    inputs = torch.randn((32, 1, 2048))
    net = RSNet(block=CWBlock)
    outputs = net(inputs)
    pass
