import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional


def convkx1(in_planes: int, out_planes: int, kernel_size: int = 3, stride: int = 1) -> nn.Conv1d:
    """3x1 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size - 1) // 2)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv1d:
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = convkx1(inplanes, planes, 3, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convkx1(planes, planes, 3)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = 1
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = convkx1(width, width, kernel_size, stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, inplanes)
        self.bn3 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        xk = out

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out, xk


class TFCNN(nn.Module):
    def __init__(
            self,
            num_classes: int = 10,
            norm_layer: Optional[Callable[..., nn.Module]] = None
            ):
        super(TFCNN, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        
        self.inplanes = 1
        # TFCNN V1
        # self.layer1 = self._make_layer(16, kernel_size=29)
        # self.layer2 = self._make_layer(32, kernel_size=29)
        # self.layer3 = self._make_layer(64, kernel_size=61)
        # self.layer4 = self._make_layer(128, kernel_size=125)
        # self.layer5 = self._make_layer(256, kernel_size=253)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(256, num_classes)

        # TFCNN V2
        self.layer1 = self._make_layer(64, kernel_size=13)
        self.layer2 = self._make_layer(64, kernel_size=45)
        self.layer3 = self._make_layer(64, kernel_size=61)
        self.layer4 = self._make_layer(64, kernel_size=125)
        self.layer5 = self._make_layer(64, kernel_size=253)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)
        
        # TFCNN V3
        # self.layer1 = self._make_layer(64, kernel_size=13)
        # self.layer2 = self._make_layer(64, kernel_size=13)
        # self.layer3 = self._make_layer(64, kernel_size=29)
        # self.layer4 = self._make_layer(64, kernel_size=61)
        # self.layer5 = self._make_layer(64, kernel_size=125)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(64, num_classes)

        # TFCNN V4
        # self.layer1 = self._make_layer(64, kernel_size=3)
        # self.layer2 = self._make_layer(64, kernel_size=3)
        # self.layer3 = self._make_layer(64, kernel_size=3)
        # self.layer4 = self._make_layer(64, kernel_size=3)
        # self.layer5 = self._make_layer(64, kernel_size=3)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(64, num_classes)

    
    def _make_layer(self, planes, stride=1, kernel_size=13):
        layers = []
        if self.inplanes != planes:
            layers.append(conv1x1(self.inplanes, planes))
            self.inplanes = planes
        layers.append(BasicBlock(planes, planes, downsample=None))
        layers.append(Bottleneck(planes, planes, kernel_size, stride, downsample=None))
        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        x, x1 = self.layer1(x)
        x, x2 = self.layer2(x)
        x, x3 = self.layer3(x)
        x, x4 = self.layer4(x)
        x, x5 = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, [x1, x2, x3, x4, x5]

if __name__ == "__main__":
    x = torch.randn(1, 1, 1000)
    model = TFCNN(num_classes=4)
    y, features = model(x)
