# ultralytics/nn/modules/yola.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectedConvolution(nn.Module):
    def __init__(self, kernel_nums: int = 8, kernel_size: int = 3):
        super().__init__()
        self.kernel_nums = kernel_nums
        self.kernel_size = kernel_size

        self.rg_bn = nn.BatchNorm2d(kernel_nums)
        self.gb_bn = nn.BatchNorm2d(kernel_nums)
        self.rb_bn = nn.BatchNorm2d(kernel_nums)

        self.filter = nn.Parameter(torch.randn(self.kernel_nums, 1, self.kernel_size, self.kernel_size))
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.filter)
        nn.init.constant_(self.rg_bn.weight, 0.01)
        nn.init.constant_(self.rg_bn.bias, 0)
        nn.init.constant_(self.gb_bn.weight, 0.01)
        nn.init.constant_(self.gb_bn.bias, 0)
        nn.init.constant_(self.rb_bn.weight, 0.01)
        nn.init.constant_(self.rb_bn.bias, 0)

    @staticmethod
    def mean_constraint(kernel: torch.Tensor) -> torch.Tensor:
        bs, cin, kw, kh = kernel.shape
        kernel_mean = torch.mean(kernel.view(bs, -1), dim=1, keepdim=True)
        kernel = kernel.view(bs, -1) - kernel_mean
        return kernel.view(bs, cin, kw, kh)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        img: (B, 3, H, W), expected in [0, 1]
        return: (B, 3 * kernel_nums, H, W)
        """
        zero_masks = torch.zeros_like(img)
        zero_masks[img == 0] = 1

        log_img = torch.log(img.clamp_min(1e-7))

        red_chan = log_img[:, 0:1, :, :]
        green_chan = log_img[:, 1:2, :, :]
        blue_chan = log_img[:, 2:3, :, :]

        normalized_filter = self.mean_constraint(self.filter)

        # Red-Green
        filt_r1 = F.conv2d(red_chan, weight=normalized_filter, padding=self.kernel_size // 2)
        filt_g1 = F.conv2d(green_chan, weight=-normalized_filter, padding=self.kernel_size // 2)
        filt_rg = self.rg_bn(filt_r1 + filt_g1)

        # Green-Blue
        filt_g2 = F.conv2d(green_chan, weight=normalized_filter, padding=self.kernel_size // 2)
        filt_b1 = F.conv2d(blue_chan, weight=-normalized_filter, padding=self.kernel_size // 2)
        filt_gb = self.gb_bn(filt_g2 + filt_b1)

        # Red-Blue
        filt_r2 = F.conv2d(red_chan, weight=normalized_filter, padding=self.kernel_size // 2)
        filt_b2 = F.conv2d(blue_chan, weight=-normalized_filter, padding=self.kernel_size // 2)
        filt_rb = self.rb_bn(filt_r2 + filt_b2)

        rg = torch.where(zero_masks[:, 0:1, ...].expand(-1, self.kernel_nums, -1, -1) == 1, 0, filt_rg)
        gb = torch.where(zero_masks[:, 1:2, ...].expand(-1, self.kernel_nums, -1, -1) == 1, 0, filt_gb)
        rb = torch.where(zero_masks[:, 2:3, ...].expand(-1, self.kernel_nums, -1, -1) == 1, 0, filt_rb)

        return torch.cat([rg, gb, rb], dim=1)


class IIBlock(nn.Module):
    def __init__(self, kernel_nums: int = 8, kernel_size: int = 3, Gtheta=(0.6, 0.8)):
        super().__init__()
        self.Gtheta = Gtheta
        self.iim = ReflectedConvolution(kernel_nums, kernel_size)

        self.feat_projector = nn.Sequential(
            nn.Conv2d(3, 24, 3, 1, 1, groups=1, bias=False),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(inplace=True),
        )

        self.fuse_net = nn.Sequential(
            nn.Conv2d(48, 32, 3, 1, 1, groups=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 3, 3, 1, 1, groups=1, bias=True),
        )

        self.aux_feats = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_gma = torch.pow(x, float(np.random.uniform(self.Gtheta[0], self.Gtheta[1]))).clamp(0, 1)

        feat_ii = self.iim(x)
        feat_ii_gma = self.iim(x_gma)

        feats = self.feat_projector(x)
        x_out = self.fuse_net(torch.cat((feats, feat_ii), dim=1))

        self.aux_feats = (feat_ii, feat_ii_gma)
        return x_out


class YOLAConv(nn.Module):
    """
    Custom first stage for YOLOv8:
    IIBlock -> standard Conv to c2 channels.
    Signature is designed to be friendly to Ultralytics parse_model.
    """
    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 2,
        kernel_nums: int = 8,
        kernel_size: int = 3,
        Gtheta=(0.6, 0.8),
    ):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

        self.ii = IIBlock(kernel_nums=kernel_nums, kernel_size=kernel_size, Gtheta=Gtheta)
        self.conv = nn.Sequential(
            nn.Conv2d(3, c2, k, s, padding=k // 2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

        self.aux_feats = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ii(x)
        self.aux_feats = self.ii.aux_feats
        return self.conv(x)