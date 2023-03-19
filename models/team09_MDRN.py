import torch
from matplotlib import pyplot as plt
from torch import nn
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.init import trunc_normal_

def get_local_weights(residual, ksize, padding):
    pad = padding
    residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')
    unfolded_residual = residual_pad.unfold(2, ksize, 3).unfold(3, ksize, 3)
    pixel_level_weight = torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)

    return pixel_level_weight


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


class PixelShuffleDirect(nn.Module):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        super(PixelShuffleDirect, self).__init__()
        self.upsampleOneStep = UpsampleOneStep(scale, num_feat, num_out_ch, input_resolution=None)

    def forward(self, x):
        return self.upsampleOneStep(x)


class DepthWiseConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_norm=False, bn_kwargs=None):
        super(DepthWiseConv, self).__init__()

        self.dw = torch.nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_ch,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.pw = torch.nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

    def forward(self, input):
        out = self.dw(input)
        out = self.pw(out)
        return out


class BSConvU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, padding_mode="zeros", with_ln=False, bn_kwargs=None):
        super().__init__()

        # pointwise
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
        )

        # depthwise
        self.dw = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding_mode='reflect',
        )

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=14):
        super(CCALayer, self).__init__()
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class Dconv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, (1, 1), padding=padding)
        self.conv2 = nn.Conv2d(out_dim, out_dim, (kernel_size, kernel_size), padding=padding, groups=out_dim)

    def forward(self, input):
        out = self.conv2(self.conv1(input))
        return out

class Conv_Gelu_Res(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding='same'):
        super().__init__()
        self.conv1 = Dconv(in_dim, out_dim, kernel_size, padding)
        self.act = nn.GELU()

    def forward(self, input):
        out = self.act(self.conv1(input) + input)
        return out


class CA(nn.Module):
    def __init__(self, c_dim, reduction):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c_dim, c_dim, (1, 1), padding='same'),
                                  nn.GELU(),
                                  CCALayer(c_dim, reduction),
                                  nn.Conv2d(c_dim, c_dim, (3, 3), padding='same', groups=c_dim))

    def forward(self, x):
        ca_x = self.body(x)
        ca_x += x
        return ca_x


class SA(nn.Module):
    def __init__(self, c_dim, conv):
        super().__init__()
        self.body = nn.Sequential(ESA(c_dim, conv))

    def forward(self, x):
        sa_x = self.body(x)
        sa_x += x
        return sa_x


class ESA(nn.Module):
    def __init__(self, num_feat=50, conv=nn.Conv2d, p=0.25):
        super(ESA, self).__init__()
        f = num_feat // 4
        BSConvS_kwargs = {}
        if conv.__name__ == 'BSConvS':
            BSConvS_kwargs = {'p': p}
        self.conv1 = nn.Conv2d(num_feat, f, 1)
        self.conv_f = nn.Conv2d(f, f, 1)
        self.conv2_0 = conv(f, f, 3, 2, 1, padding_mode='reflect')
        self.conv2_1 = conv(f, f, 3, 2, 1, padding_mode='reflect')
        self.conv2_2 = conv(f, f, 3, 2, 1, padding_mode='reflect')
        self.conv2_3 = conv(f, f, 3, 2, 1, padding_mode='reflect')
        self.maxPooling_0 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        self.maxPooling_1 = nn.MaxPool2d(kernel_size=5, stride=3)
        self.maxPooling_2 = nn.MaxPool2d(kernel_size=7, stride=3, padding=1)
        self.conv_max_0 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv_max_1 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv_max_2 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.var_3 = get_local_weights
        self.conv3_0 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv3_1 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv3_2 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv4 = nn.Conv2d(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()
        # self.norm = nn.BatchNorm2d(num_feat)
        # self.seita = nn.Parameter(torch.normal(mean=0.5, std=0.01, size=(1, 1, 1)))
        # self.keci = nn.Parameter(torch.normal(mean=0.5, std=0.01, size=(1, 1, 1)))
        #
        # self.alpha = nn.Parameter(torch.normal(mean=0.25, std=0.01, size=(1,1,1)))
        # self.beta = nn.Parameter(torch.normal(mean=0.25, std=0.01, size=(1,1,1)))
        # self.gama = nn.Parameter(torch.normal(mean=0.25, std=0.01, size=(1,1,1)))
        # self.omega = nn.Parameter(torch.normal(mean=0.25, std=0.01, size=(1,1,1)))

    def forward(self, input):
        c1_ = self.conv1(input)
        temp = self.conv2_0(c1_)
        c1_0 = self.maxPooling_0(temp)
        c1_1 = self.maxPooling_1(self.conv2_1(c1_))
        c1_2 = self.maxPooling_2(self.conv2_2(c1_))
        c1_3 = self.var_3(self.conv2_3(c1_), 7, padding=1)
        v_range_0 = self.conv3_0(self.GELU(self.conv_max_0(c1_0)))
        v_range_1 = self.conv3_1(self.GELU(self.conv_max_1(c1_1)))
        v_range_2 = self.conv3_2(self.GELU(self.conv_max_2(c1_2 + c1_3)))
        c3_0 = F.interpolate(v_range_0, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        c3_1 = F.interpolate(v_range_1, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        c3_2 = F.interpolate(v_range_2, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4((c3_0 + c3_1 + c3_2 + cf))
        m = self.sigmoid(c4)

        return input * m


class EADB(nn.Module):
    def __init__(self, in_channels, out_channels, conv=nn.Conv2d, p=0.25):
        super(EADB, self).__init__()
        kwargs = {'padding': 1}

        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = conv(in_channels, self.rc, kernel_size=3, **kwargs)
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c2_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c3_r = conv(self.remaining_channels, self.rc, kernel_size=3, **kwargs)

        self.c4 = conv(self.remaining_channels, self.dc, kernel_size=3, **kwargs)
        self.act = nn.GELU()

        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1)
        self.esa = SA(in_channels, conv)
        self.cca = CA(in_channels, reduction=16)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)
        out_fused = self.esa(out)
        out_fused = self.cca(out_fused)
        return out_fused + input


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class MDRN(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=28, num_block=8, num_out_ch=3, upscale=4
                 , rgb_mean=(0.4488, 0.4371, 0.4040), p=0.25):
        super(MDRN, self).__init__()
        kwargs = {'padding': 1}
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.conv = BSConvU
        self.fea_conv = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding='same')

        self.B1 = EADB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B2 = EADB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B3 = EADB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B4 = EADB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B5 = EADB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B6 = EADB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B7 = EADB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        self.B8 = EADB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        # self.B9 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)
        # self.B10 = ESDB(in_channels=num_feat, out_channels=num_feat, conv=self.conv, p=p)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()
        self.c2 = self.conv(num_feat, num_feat, kernel_size=3, **kwargs)
        self.upsampler = PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)

    def forward(self, input):
        self.mean = self.mean.type_as(input)
        input = input - self.mean
        # SR
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)

        out = self.upsampler(self.c2(self.GELU(self.c1(
            torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6, out_B7, out_B8], dim=1)))) + out_fea)\
              + self.mean
        # Denoise
        # out_fea = self.fea_conv(denosed_input)
        # out_B1 = self.B1(out_fea)
        # out_B2 = self.B2(out_B1)
        # out_B3 = self.B3(out_B2)
        # out_B4 = self.B4(out_B3)
        # out_B5 = self.B5(out_B4)
        # # out_B6 = self.B6(out_B5)
        # # out_B7 = self.B7(out_B6)
        # # out_B8 = self.B8(out_B7)
        # out = self.c2(self.GELU(self.c1(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5], dim=1)))) + out_fea
        # if detach_ture:
        #     output_denosed = self.to_RGB(out.detach()) + self.mean
        # else:
        #     output_denosed = self.to_RGB(out) + self.mean
        return out

