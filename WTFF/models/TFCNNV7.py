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
        uk: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        upsample: Optional[int] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        width = 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = convkx1(width, width, kernel_size, stride)
        self.bn2 = norm_layer(width)
        self.upsample = None
        if upsample: self.upsample = nn.ConvTranspose1d(1, 1, kernel_size=uk, stride=upsample, padding=upsample//2)
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

        if self.upsample: xk = self.upsample(xk)

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
        
        self.inplanes = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # TFCNN V6
        self.layer1 = self._make_layer(64, kernel_size=3, stride=2)
        self.layer2 = self._make_layer(64, kernel_size=3)
        self.layer3 = self._make_layer(128, kernel_size=3, stride=2, upsample=2, uk=4)
        self.layer4 = self._make_layer(128, kernel_size=3, upsample=2, uk=4)
        self.layer5 = self._make_layer(256, kernel_size=3, stride=2, upsample=4, uk=8)
        self.layer6 = self._make_layer(256, kernel_size=3, upsample=4, uk=8)

        # TFCNN V6 C1
        # self.layer1 = self._make_layer(64, kernel_size=5, stride=2)
        # self.layer2 = self._make_layer(64, kernel_size=5)
        # self.layer3 = self._make_layer(128, kernel_size=5, stride=2, upsample=2, uk=4)
        # self.layer4 = self._make_layer(128, kernel_size=5, upsample=2, uk=4)
        # self.layer5 = self._make_layer(256, kernel_size=5, stride=2, upsample=4, uk=8)
        # self.layer6 = self._make_layer(256, kernel_size=5, upsample=4, uk=8)


        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, planes, stride=1, kernel_size=13, upsample=None, uk=1):
        layers = []
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm1d(planes),
            )

        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        layers.append(Bottleneck(planes, planes, kernel_size, upsample=upsample, uk=uk))
        self.inplanes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x, x1 = self.layer1(x)
        x, x2 = self.layer2(x)
        x, x3 = self.layer3(x)
        x, x4 = self.layer4(x)
        x, x5 = self.layer5(x)
        x, x6 = self.layer6(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, [x1, x2, x3, x4, x5, x6]

if __name__ == "__main__":
    x = torch.randn(1, 1, 2048)
    model = TFCNN(num_classes=4)
    y, features = model(x)
