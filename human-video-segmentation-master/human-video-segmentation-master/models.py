from typing import *
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_relu_conv(in_channels: int,
                    out_channels: int,
                    kernel_size: int,
                    stride: int = 1,
                    padding: int = 0,
                    num: int = 1):
    layers = []
    for i in range(num):
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
    return nn.Sequential(*layers)


class ConvX(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 bias: bool = True,
                 activator=nn.ReLU,
                 groups: int = 1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                              bias=bias, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = activator(inplace=True)
        _init_weight(self)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ZoomedConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 out_ratio: float = 1,
                 inter_ratio: float = 0.5,
                 num_conv: int = 1,
                 mode: str = 'bilinear'):
        super(ZoomedConv, self).__init__()
        self.inter_ratio = inter_ratio
        self.out_ratio = out_ratio / inter_ratio
        self.mode = mode
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.other_convs = _make_relu_conv(out_channels, out_channels, 3, 1, 1, num_conv - 1)
        _init_weight(self)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.inter_ratio, mode=self.mode)
        x = self.conv1(x)
        x = self.other_convs(x)
        x = F.interpolate(x, scale_factor=self.out_ratio, mode=self.mode)
        return x


class ZConvX(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 out_ratio: float = 1,
                 inter_ratio: float = 0.5,
                 num_conv: int = 1,
                 mode: str = 'bilinear',
                 activator=nn.ReLU):
        super(ZConvX, self).__init__()
        self.conv = ZoomedConv(in_channels, out_channels, out_ratio, inter_ratio, num_conv, mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = activator(inplace=True)
        _init_weight(self)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CatConvX(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 down_smaple: bool = False,
                 num_conv: int = 1,
                 activator=nn.ReLU,
                 use_se: bool = False):
        super(CatConvX, self).__init__()
        assert (out_channels % 2 == 0)

        self.use_se = use_se

        if down_smaple:
            self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 3, 2, 1)
            self.conv2 = ZoomedConv(in_channels, out_channels // 2, 0.5, 0.25, num_conv)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 3, 1, 1)
            self.conv2 = ZoomedConv(in_channels, out_channels // 2, 1, 0.5, num_conv)
        self.other_convs1 = _make_relu_conv(out_channels // 2, out_channels // 2, 3, 1, 1, num_conv - 1)
        if use_se:
            self.se = SqueezeExcitation(out_channels // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = activator(True)
        _init_weight(self)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.other_convs1(x1)
        x2 = self.conv2(x)
        x = torch.cat((x1, x2), 1)
        if self.use_se:
            x = self.se(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SqueezeExcitation(nn.Module):
    def __init__(self,
                 in_channels: int,
                 squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        sqz_channels = _make_divisible(in_channels // squeeze_factor, 8)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_channels, sqz_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(sqz_channels, in_channels, 1)
        self.sigmoid = nn.Hardsigmoid(inplace=True)
        _init_weight(self)

    def forward(self, x):
        atten = self.pool(x)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        x = torch.mul(atten, x)
        return x


class ARM(nn.Module):
    def __init__(self, num_channels: int):
        super(ARM, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(num_channels, num_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(num_channels)
        self.sigmoid = nn.Hardsigmoid(inplace=True)
        _init_weight(self)

    def forward(self, x):
        atten = self.pool(x)
        atten = self.conv(atten)
        atten = self.bn(atten)
        atten = self.sigmoid(atten)
        x = torch.mul(atten, x)
        return x


class ARMPlus(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ARMPlus, self).__init__()
        self.conv = ConvX(in_channels, out_channels, 3, 1, 1)
        self.arm = ARM(out_channels)
        _init_weight(self)

    def forward(self, x):
        x = self.conv(x)
        x = self.arm(x)
        return x


class FFM(nn.Module):
    def __init__(self,
                 in_channels1: int,
                 in_channels2: int,
                 out_channels: int,
                 inter_ratio: int = 4):
        super(FFM, self).__init__()
        in_channels = in_channels1 + in_channels2
        inter_channels = _make_divisible(out_channels // inter_ratio, 8)

        self.convx = ConvX(in_channels, out_channels, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(out_channels, inter_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, out_channels, 1, bias=False)
        self.sigmoid = nn.Hardsigmoid(inplace=True)
        _init_weight(self)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.convx(x)
        atten = self.pool(x)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        y = torch.mul(x, atten)
        x = torch.add(x, y)
        return x


class GPoolInterpolateConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(GPoolInterpolateConv, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = ConvX(in_channels, out_channels, 1)

    def forward(self, x):
        shape = x.size()[2:]
        x = self.pool(x)
        x = F.interpolate(x, shape)
        x = self.conv(x)
        return x


class CatBottleneck(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 down_sample: bool = True):
        super(CatBottleneck, self).__init__()
        self.down_sample = down_sample
        self.block1 = ConvX(in_channels, out_channels // 2, 1)

        if self.down_sample:
            self.block2 = ConvX(out_channels // 2, out_channels // 4, 3, 2, 1)
            self.pool = nn.AvgPool2d(3, 2, 1)
        else:
            self.block2 = ConvX(out_channels // 2, out_channels // 4, 3, 1, 1)

        self.block3 = ConvX(out_channels // 4, out_channels // 8, 3, 1, 1)
        self.block4 = ConvX(out_channels // 8, out_channels // 8, 3, 1, 1)
        _init_weight(self)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        if self.down_sample:
            x1 = self.pool(x1)
        x = torch.cat((x1, x2, x3, x4), 1)
        return x


class InvertedResidual(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 inner_channels: int,
                 down_sample: bool = False,
                 activator=nn.Hardswish,
                 use_se: bool = True):
        super(InvertedResidual, self).__init__()

        self.shortcut = in_channels == out_channels and not down_sample

        layers = []

        if inner_channels != in_channels:
            layers.append(ConvX(in_channels, inner_channels, 1, 1, activator=activator))

        if down_sample:
            layers.append(ConvX(inner_channels, inner_channels, 3, 2, 1, activator=activator))
        else:
            layers.append(ConvX(inner_channels, inner_channels, 3, 1, 1, activator=activator))

        if use_se:
            layers.append(SqueezeExcitation(inner_channels, 4))
        layers.append(ConvX(inner_channels, out_channels, 1, activator=nn.Identity))

        self.layers = nn.Sequential(*layers)
        _init_weight(self)

    def forward(self, x):
        y = self.layers(x)
        if self.shortcut:
            y += x
        return y


class STDCStage(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_necks: int = 2):
        super(STDCStage, self).__init__()
        self.neck1 = CatBottleneck(in_channels, out_channels, True)
        self.other_necks = nn.Sequential(
            *[CatBottleneck(out_channels, out_channels, False) for _ in range(num_necks - 1)]
        )
        _init_weight(self)

    def forward(self, x):
        x = self.neck1(x)
        x = self.other_necks(x)
        return x


class UpPredict(nn.Module):
    def __init__(self,
                 num_channels: int,
                 num_classes: int,
                 inter_channels: int,
                 out_ratio: float,
                 mode: str = 'bilinear'):
        super(UpPredict, self).__init__()
        self.out_ratio = out_ratio
        self.mode = mode
        self.convx = ConvX(num_channels, inter_channels, 3, 1, 1)
        self.conv = nn.Conv2d(inter_channels, num_classes, 1, bias=False)
        _init_weight(self)

    def forward(self, x):
        x = self.convx(x)
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.out_ratio, mode=self.mode)
        return x


class MobileUpSample(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 up_ratio: float,
                 mode: str = 'bilinear'):
        super(MobileUpSample, self).__init__()
        self.mode = mode
        self.up_ratio = up_ratio

    def forward(self, x):
        x = self.block1(x)
        x = F.interpolate(x, scale_factor=self.up_ratio, mode=self.mode)
        return x


class UpPredictV2(nn.Module):
    def __init__(self,
                 in_channels1: int,
                 in_channels2: int,
                 in_channels3: int,
                 out_channels: int,
                 mode: str = 'bilinear'):
        super(UpPredictV2, self).__init__()

        self.mode = mode
        in_channels = in_channels1 + in_channels2 + in_channels3
        self.pool = nn.AvgPool2d(3, 2, 1)
        self.conv = ConvX(in_channels, out_channels, 3, 1, 1)
        _init_weight(self)

    def forward(self, x1, x2, x3):
        x3 = self.pool(x3)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=2, mode=self.mode)
        return x

class UpPredictV2_2(nn.Module):
    def __init__(self,
                 in_channels1: int,
                 in_channels2: int,
                 out_channels: int,
                 mode: str = 'bilinear'):
        super(UpPredictV2_2, self).__init__()

        self.mode = mode
        in_channels = in_channels1 + in_channels2
        self.conv = ConvX(in_channels, out_channels, 3, 1, 1)
        _init_weight(self)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=2, mode=self.mode)
        return x

class LastConv(nn.Module):
    def __init__(self,
                 in_channels1: int,
                 in_channels2: int,
                 out_channels: int,
                 inner_channels: int):
        super(LastConv, self).__init__()

        in_channels = in_channels1 + in_channels2
        self.convs = nn.Sequential(
            ConvX(in_channels, inner_channels, 3, 1, 1),
            nn.Conv2d(inner_channels, out_channels, 3, 1, 1)
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.convs(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        return x
