from typing import *
from models import *
from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F


# down sample to 8x
class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        self.conv1 = CatConvX(3, 16, down_smaple=True)
        self.conv2 = CatConvX(16, 32, down_smaple=True, num_conv=2)
        self.conv3 = CatConvX(32, 64, down_smaple=True, num_conv=2)
        # self.conv3 = STDCStage(32, 64)
        _init_weight(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class MobileStem(nn.Module):
    def __init__(self):
        super(MobileStem, self).__init__()
        self.conv1 = ConvX(3, 16, 3, 2, 1, bias=False, activator=nn.Hardswish)
        self.block1 = InvertedResidual(16, 16, 16, down_sample=False, activator=nn.ReLU, use_se=False)
        self.block2 = InvertedResidual(16, 24, 64, down_sample=True, activator=nn.ReLU, use_se=False)
        self.block3 = InvertedResidual(24, 24, 72, down_sample=False, activator=nn.ReLU, use_se=False)
        self.block4 = InvertedResidual(24, 64, 96, down_sample=True, activator=nn.ReLU)
        _init_weight(self)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
        self.stage1 = STDCStage(64, 128)  # down sample to 16x
        self.stage2 = STDCStage(128, 256)  # down sample to 32x
        _init_weight(self)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        return x1, x2


# fusion and up sample to 8x
class ARMFusion(nn.Module):
    def __init__(self):
        super(ARMFusion, self).__init__()
        self.arm16 = ARMPlus(128, 64)
        # self.conv16 = ConvX(64, 64, 3, 1, 1)
        self.conv16 = CatConvX(64, 64, False)

        self.arm32 = ARMPlus(256, 64)
        self.pool32 = GPoolInterpolateConv(256, 64)
        self.conv32 = ConvX(64, 64, 3, 1, 1)
        _init_weight(self)

    def forward(self, x16, x32):
        x32_arm = self.arm32(x32)
        x32_arm += self.pool32(x32)
        x32_arm = F.interpolate(x32_arm, scale_factor=2)
        x32_arm = self.conv32(x32_arm)

        x16_arm = self.arm16(x16)
        x16_arm += x32_arm
        x16_arm = F.interpolate(x16_arm, scale_factor=2)
        x16_arm = self.conv16(x16_arm)
        return x16_arm  # 8x


class HVSNet(nn.Module):
    def __init__(self,
                 num_classes: int = 2,
                 mode: str = 'eval',
                 stem: str = 'original',
                 type: str = 'segmentation'):
        super(HVSNet, self).__init__()
        assert mode in ('eval', 'train')
        assert stem in ('original', 'mobilenet_v3')
        assert type in ('segmentation', 'matting')

        self.mode = mode
        self.type = type
        self.version = 'v1'

        if stem == 'original':
            self.stem = Stem()
        else:
            self.stem = MobileStem()

        self.context = ContextPath()
        self.fusion = ARMFusion()
        self.ffm = FFM(64, 64, 64)

        if type == 'segmentation':
            self.up = UpPredict(64, num_classes, 32, 8)
            # self.up1 = UpPredict(64, 16, 32, 2)
            # self.up2 = UpPredict(16, num_classes, 4, 4)

            if mode == 'train':
                self.detail_up = UpPredict(64, 3, 32, 8)

        elif type == 'matting':
            # self.up = UpPredict(64, 1, 32, 8)
            self.up1 = UpPredict(64, 16, 32, 4)
            self.up2 = UpPredict(16, 1, 8, 2)

        _init_weight(self)

    def forward(self, x):
        if self.type == 'segmentation':
            return self._forward_impl_segmentation(x)
        elif self.type == 'matting':
            return self._forward_impl_matting(x)

    def _forward_impl_segmentation(self, x):
        x8_1 = self.stem(x)  # detail output
        x16, x32 = self.context(x8_1)  # context output
        x8_2 = self.fusion(x16, x32)
        x = self.ffm(x8_1, x8_2)
        x = self.up(x)
        # x = self.up1(x)
        # x = self.up2(x)

        if self.mode == 'train':
            detail_x = self.detail_up(x8_1)
            return x, detail_x

        return x

    def _forward_impl_matting(self, x):
        x8_1 = self.stem(x)
        x16, x32 = self.context(x8_1)
        x8_2 = self.fusion(x16, x32)
        x = self.ffm(x8_1, x8_2)
        x = self.up1(x)
        x = self.up2(x)
        # x = self.up(x)
        return x.squeeze(1)


#################################

class StemV2(nn.Module):
    def __init__(self):
        super(StemV2, self).__init__()
        self.conv1 = CatConvX(3, 16, down_smaple=True)
        self.conv2 = CatConvX(16, 32, down_smaple=True)
        self.conv3 = CatConvX(32, 64, down_smaple=True)
        _init_weight(self)

    def forward(self, x):
        x2 = self.conv1(x)
        x4 = self.conv2(x2)
        x8 = self.conv3(x4)
        return x, x2, x4, x8


class HVSNetV2(nn.Module):
    def __init__(self, type='segmentation'):
        super(HVSNetV2, self).__init__()
        assert type in ('segmentation', 'matting')
        self.type = type
        self.version = 'v2'

        out_channels = 2 if type == 'segmentation' else 1

        self.stem = StemV2()
        self.context = ContextPath()
        self.fusion = ARMFusion()
        self.ffm = FFM(64, 64, 32)

        self.up_4x_2x = UpPredictV2(32, 32, 16, 16)
        self.up_2x_1x = LastConv(16, 16, out_channels, 4)


    def forward(self, x):
        x, x2, x4, x8 = self.stem(x)

        x16, x32 = self.context(x8)  # context output
        x8_2 = self.fusion(x16, x32)
        x8_fusion = self.ffm(x8, x8_2)
        x4_2 = F.interpolate(x8_fusion, scale_factor=2, mode='bilinear')

        x2_2 = self.up_4x_2x(x4_2, x4, x2)
        x = self.up_2x_1x(x2_2, x2)
        return x.squeeze(1)
