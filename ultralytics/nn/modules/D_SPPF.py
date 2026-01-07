import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad


__all__ = (
    "D_SPPF",
)


class DynamicPooling(nn.Module):
    def __init__(self, in_channels, pool_kernels=[3, 5, 7]):
        super().__init__()
        self.pool_kernels = pool_kernels
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, len(pool_kernels)),
            nn.Sigmoid()
        )
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) for k in pool_kernels
        ])

    def forward(self, x):
        B, C, H, W = x.size()
        context = self.global_pool(x).view(B, C)
        weights = self.fc(context)
        pooled_outs = []
        for i, pool in enumerate(self.pools):
            pooled = pool(x)
            pooled_outs.append(pooled * weights[:, i].view(B, 1, 1, 1))
        return sum(pooled_outs)


class D_SPPF(nn.Module):
    def __init__(self, c1, c2, base_pool_kernel=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, k=1, s=1)
        self.dynamic_pool = DynamicPooling(c_, pool_kernels=[base_pool_kernel - 2, base_pool_kernel, base_pool_kernel + 2])
        self.std_pool = nn.MaxPool2d(kernel_size=base_pool_kernel, stride=1, padding=base_pool_kernel // 2)
        self.attn = CoordAtt(c_ * 2)
        self.cv2 = Conv(c_ * 2, c2, k=1, s=1)

    def forward(self, x):
        x = self.cv1(x)
        dpool = self.dynamic_pool(x)
        spool = self.std_pool(x)
        out = torch.cat((dpool, spool), dim=1)
        out = self.attn(out)
        out = self.cv2(out)
        return out


class CoordAtt(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, c, h, w = x.size()
        x_h = self.pool_h(x)  # (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, W, 1)
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+W, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return x * a_h * a_w