from __future__ import absolute_import

import torch.nn as nn
import torch

class ConvNet(nn.Module):

    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.inplanes = 4

        self.conv1 = nn.Conv1d(1, 4, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(4)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(4, 4, stride=2)
        self.layer2 = self._make_layer(8, 4, stride=2)
        self.layer3 = self._make_layer(16, 4, stride=2)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm1d(planes),
            )

        layers = []
        layers.append(downsample)
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=3, padding=1),
                nn.BatchNorm1d(planes),
            ))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    inputs = torch.randn((32, 1, 2048))
    net = ConvNet()
    outputs = net(inputs)
    pass
