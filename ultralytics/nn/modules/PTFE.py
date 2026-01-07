import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad


__all__ = (
    "PTFE",
)

class Polar(nn.Module):
    def __init__(self, in_channels=3):
        super(Polar, self).__init__()
        self.in_channels = in_channels
        self.bn = nn.BatchNorm2d(8)

    def forward(self, x):
        I0 = x[:, 0:1, :, :]
        I45 = x[:, 1:2, :, :]
        I90 = x[:, 2:3, :, :]
        S0 = I0 + I90
        S1 = I0 - I90
        S2 = 2 * I45 - S0
        AOP = torch.atan(S2 / (S1 + 1e-6))
        AOP = torch.where(S1 < 0, AOP + torch.pi, AOP)
        DOP = torch.sqrt(S1 ** 2 + S2 ** 2) / (S0 + 1e-6)
        concat_features = torch.cat([x, S0, S1, S2, AOP, DOP], dim=1)
        return self.bn(concat_features)


class PTFE(nn.Module):
    def __init__(self, in_channels=3):
        super(PTFE, self).__init__()

        self.polar_trans = Polar(in_channels=in_channels)
        self.conv = Conv(c1=8, c2=3, k=3)

    def forward(self, x):
        x = self.polar_trans(x)
        x = self.conv(x)
        return x