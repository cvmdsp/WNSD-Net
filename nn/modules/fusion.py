import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


class LowPassConvGenerator(nn.Module):
    def __init__(self, in_channel, num_k=2, ratio=8):
        super(LowPassConvGenerator, self).__init__()
        self.num_k = num_k
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1_list = nn.ModuleList(
            [nn.Conv2d(in_channel // num_k * 2, in_channel // ratio, 1, bias=False) for _ in range(num_k)])
        self.fc2_list = nn.ModuleList(
            [nn.Conv2d(in_channel // ratio, 49 * in_channel, 1, bias=False) for _ in range(num_k)])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.shape
        kernels = []

        for i in range(self.num_k):
            start_idx = i * (c // self.num_k)
            end_idx = (i + 1) * (c // self.num_k)
            x_i = x[:, start_idx:end_idx, :, :]

            avg_feature_x = self.avg_pool(x_i)  # b, c/num_k, 1, 1
            max_feature_x = self.max_pool(x_i)  # b, c/num_k, 1, 1

            fc1 = self.fc1_list[i]
            fc2 = self.fc2_list[i]

            combined_feature = torch.cat([avg_feature_x, max_feature_x], dim=1)
            kernel = fc2(self.relu(fc1(combined_feature)))  # (b, 9c, 1, 1)

            kernel = kernel.reshape(b, c, 49, 1)
            kernel = F.softmax(kernel, dim=2)
            kernel = kernel.reshape(b, c, 7, 7)[0].unsqueeze(1)  # (c/num_k, 1, 3, 3)

            kernels.append(kernel)
        final_kernel = torch.cat(kernels, dim=0)  # (num_k * c, 1, 3, 3)

        return final_kernel



class HighPassConvGenerator(nn.Module):
    def __init__(self, in_channel, num_k=2, kernel_size=3, ratio=8):
        super(HighPassConvGenerator, self).__init__()
        self.num_k = num_k
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1_list = nn.ModuleList(
            [nn.Conv2d(2*in_channel // num_k, in_channel // ratio, 1, bias=False) for _ in range(num_k)])
        self.fc2_list = nn.ModuleList(
            [nn.Conv2d(in_channel // ratio, 9 * in_channel // num_k, 1, bias=False) for _ in range(num_k)])
        self.relu = nn.ReLU(inplace=True)

        self.identity_kernel = nn.Parameter(torch.tensor([[0, 0, 0],
                                                          [0, 1, 0],
                                                          [0, 0, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                                            requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape  # 64,64,64
        kernels = []

        for i in range(self.num_k):
            start_idx = i * (c // self.num_k)
            end_idx = (i + 1) * (c // self.num_k)
            x_i = x[:, start_idx:end_idx, :, :]

            avg_feature_x = self.avg_pool(x_i)  # b, c/num_k, 1, 1
            max_feature_x = self.max_pool(x_i)  # b, c/num_k, 1, 1
            fc1 = self.fc1_list[i]
            fc2 = self.fc2_list[i]

            combined_feature = torch.cat([avg_feature_x, max_feature_x], dim=1)
            kernel = fc2(self.relu(fc1(combined_feature)))  # (b, 9c, 1, 1)

            kernel = kernel.reshape(b, c, 9, 1)
            kernel = F.softmax(kernel, dim=2)
            kernel = kernel.reshape(b, c, 3, 3)[0].unsqueeze(1)  # (c/num_k, 1, 3, 3)

            kernels.append(kernel)
        final_kernel = torch.cat(kernels, dim=0)  # (num_k * c, 1, 3, 3)

        return final_kernel

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

# class Freq_Fusion(nn.Module):
#     def __init__(self, c_high=64, c_low=64, c_out=64, num_k=1):
#         super(Freq_Fusion, self).__init__()
#         #assert c_high == c_low * 2, 'c_high must be 2 * c_low'
#         self.num_k = num_k
#         self.c_high = c_high
#         self.c_low = c_low
#         self.c_out = c_out
#
#         self.y_conv1 = nn.Conv2d(c_high, c_low, 1, 1)
#         self.x_conv2 = nn.Conv2d(c_low, c_low, 1, 1)
#         self.HPG_conv = nn.Sequential(nn.Conv2d(c_high, c_high, 3, 1, padding='same', groups=c_high),
#                                       nn.BatchNorm2d(c_high),
#                                       nn.ReLU(inplace=True))
#         self.LPG_conv = nn.Sequential(nn.Conv2d(c_low, c_low, 7, 1, padding='same', groups=c_low),
#                                       nn.BatchNorm2d(c_low),
#                                       nn.ReLU(inplace=True))
#         self.Cat_conv = nn.Sequential(
#             nn.Conv2d(c_high + c_low, c_out, 3, 1, padding='same', groups=c_out),
#             nn.BatchNorm2d(c_out),
#             nn.ReLU(inplace=True),
#
#         )
#         self.res_conv = nn.Conv2d(c_low, c_low, 1, 1, padding=0)
#         self.HPG = HighPassConvGenerator(c_high, num_k=self.num_k)  # (c_low, 1, 3, 3)
#         self.LPG = LowPassConvGenerator(c_low, num_k=self.num_k)  # (c_low, 1, 3, 3)
#
#         self.x_conv_fhnorm = nn.BatchNorm2d(self.c_low)
#         self.y_conv_flnorm = nn.BatchNorm2d(self.c_high)
#
#         self.plus_conv = nn.Sequential(
#             nn.Conv2d(c_low, c_out, 1, 1, padding='same'),
#             nn.BatchNorm2d(c_out),
#             nn.ReLU(inplace=True),
#         )
#         self.out_conv = nn.Sequential(
#             nn.Conv2d(c_out, c_out, 1, 1, padding='same'),
#             nn.BatchNorm2d(c_out),
#             nn.ReLU(inplace=True),
#
#         )
#         self.ca = ChannelAttention(c_out)
#         self.sa = SimAM()
#
#     def forward(self, x_low, y_high):
#         # y_high = F.interpolate(y_high, scale_factor=2, mode='bilinear', align_corners=True)
#         cat_fuse = torch.cat([x_low, y_high], dim=1)  # c_low+c_high,64,64
#         cat_fuse = self.Cat_conv(cat_fuse)
#         cat_fuse = self.ca(cat_fuse)*cat_fuse+cat_fuse
#
#         x_h = self.HPG_conv(y_high)
#         x_l = self.LPG_conv(x_low)
#         kernel_L = self.LPG(x_l)
#         kernel_H = self.HPG(x_h)  # 64,1,3,3
#
#         x_low = F.conv2d(x_low, kernel_L, padding=3, groups=kernel_L.shape[0] // self.num_k)  # 64,64,64
#         x_low = self.x_conv_fhnorm(x_low)
#         x_low = self.x_conv2(x_low)
#
#
#         y_high_up = F.conv2d(y_high, kernel_H, padding=1, groups=kernel_H.shape[0] // self.num_k)  # 64,64,64
#         y_high_up = self.y_conv_flnorm(y_high_up)
#         y_high_up = self.y_conv1(y_high_up)
#
#         plus_fuse = x_low + y_high_up
#         plus_fuse = self.plus_conv(plus_fuse)
#         fuse = plus_fuse + cat_fuse
#         fuse = self.sa(fuse)+fuse
#
#
#         return fuse


class DMlp(nn.Module):
    def __init__(self, dim, growth_rate=2.0):
        super().__init__()
        hidden_dim = int(dim * growth_rate)  # 计算隐藏层的维度，growth_rate 控制增长比例
        # 定义卷积层序列，首先对输入做深度卷积，再进行逐点卷积
        self.conv_0 = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),  # 深度卷积，逐通道卷积
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0)  # 逐点卷积，通道数不变
        )
        self.act = nn.GELU()  # GELU 激活函数，非线性变换
        self.conv_1 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)  # 最终卷积，输出通道数为 dim

    def forward(self, x):
        x = self.conv_0(x)  # 通过卷积层 conv_0
        x = self.act(x)  # 激活函数处理
        x = self.conv_1(x)  # 通过 conv_1
        return x  # 返回处理后的特征

# SMFA 类：包含了多种特征融合与处理的操作
class SMFA(nn.Module):
    def __init__(self, dim=36):
        super(SMFA, self).__init__()
        # 1x1 卷积，增加通道数，将通道数从 dim 扩展为 dim * 2
        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        # 1x1 卷积，保持通道数为 dim
        self.linear_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        # 1x1 卷积，保持通道数为 dim
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        # 引入 DMlp 模块，用于进行特征处理
        self.lde = DMlp(dim, 2)
        # 深度卷积，通道数不变，进行空间卷积处理
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        # GELU 激活函数
        self.gelu = nn.GELU()
        self.down_scale = 8  # 下采样比例因子
        # alpha 和 belt 是可学习的参数，控制模型的加权
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))  # 乘法因子
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))  # 加法因子

    def forward(self, f):
        # 获取输入张量的尺寸：batch_size, channels, height, width
        _, _, h, w = f.shape
        # 将输入通过卷积层 linear_0 分割成 y 和 x 两部分
        y, x = self.linear_0(f).chunk(2, dim=1)
        # 对 x 部分进行下采样，并使用深度卷积处理
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        # 计算 x 的方差，作为额外的统计信息
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        # 对 x 部分做加权和调整，并通过线性卷积进行处理
        x_l = x * F.interpolate(self.gelu(self.linear_1(x_s * self.alpha + x_v * self.belt)), size=(h, w), mode='nearest')
        # 对 y 部分通过 DMlp 进行处理
        y_d = self.lde(y)
        # 将处理后的 x_l 和 y_d 进行加和，并通过 linear_2 进行最终卷积
        return self.linear_2(x_l + y_d)

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6



class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange
import os

sys.path.append(os.getcwd())


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class CAFMAttention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(CAFMAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3,
                                    bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True,
                                  groups=dim // self.num_heads, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        # local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)  # B, C, H, W
        out_conv = out_conv.squeeze(2)

        # global SA
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv

        return output


class Freq_Fusion(nn.Module):
    def __init__(self, c_high=128, c_low=64, c_out=64, num_k=1):
        super(Freq_Fusion, self).__init__()
        #assert c_high == c_low * 2, 'c_high must be 2 * c_low'
        self.num_k = num_k
        self.c_high = c_high
        self.c_low = c_low
        self.c_out = c_out

        self.down=nn.Conv2d(c_high, c_high, 3, 2, 1, groups=c_high)

        self.y_conv1 = nn.Conv2d(c_high, c_low, 1, 1)
        self.x_conv2 = nn.Conv2d(c_low, c_low, 1, 1)
        self.HPG_conv = nn.Sequential(nn.Conv2d(c_high, c_high, 3, 1, padding='same', groups=c_high),
                                      nn.BatchNorm2d(c_high),
                                      nn.ReLU(inplace=True))
        self.LPG_conv = nn.Sequential(nn.Conv2d(c_low, c_low, 7, 1, padding='same', groups=c_low),
                                      nn.BatchNorm2d(c_low),
                                      nn.ReLU(inplace=True))
        self.Cat_conv = nn.Sequential(
            nn.Conv2d(c_high//4 + c_low, c_out, 3, 1, padding='same', groups=c_high//4),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),

        )
        self.res_conv = nn.Conv2d(c_low, c_low, 1, 1, padding=0)
        self.HPG = HighPassConvGenerator(c_high, num_k=self.num_k)  # (c_low, 1, 3, 3)
        self.LPG = LowPassConvGenerator(c_low, num_k=self.num_k)  # (c_low, 1, 3, 3)

        self.x_conv_fhnorm = nn.BatchNorm2d(self.c_low)
        self.y_conv_flnorm = nn.BatchNorm2d(self.c_high)

        self.plus_conv = nn.Sequential(
            nn.Conv2d(c_low, c_out, 1, 1, padding='same'),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(2*c_out, c_out, 1, 1, padding='same'),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),

        )


        self.sa = CAFMAttention(c_out)

    def forward(self, x_low, y_high):
        y_high=nn.PixelShuffle(2)(y_high)
        # x_low = nn.PixelShuffle(2)(x_low)

        #y_high = self.down(y_high)
        #y_high = F.interpolate(y_high, scale_factor=2, mode='bilinear', align_corners=True)
        cat_fuse = torch.cat([x_low, y_high], dim=1)  # c_low+c_high,64,64
        cat_fuse = self.Cat_conv(cat_fuse)
        fuse = self.sa(cat_fuse)+cat_fuse

        # #
        # x_h = self.HPG_conv(y_high)
        # x_l = self.LPG_conv(x_low)
        # kernel_L = self.LPG(x_l)
        # kernel_H = self.HPG(x_h)  # 64,1,3,3
        #
        # x_low = F.conv2d(x_low, kernel_L, padding=3, groups=kernel_L.shape[0] // self.num_k)  # 64,64,64
        # x_low = self.x_conv_fhnorm(x_low)
        # x_low = self.x_conv2(x_low)
        #
        #
        # y_high_up = F.conv2d(y_high, kernel_H, padding=1, groups=kernel_H.shape[0] // self.num_k)  # 64,64,64
        # y_high_up = self.y_conv_flnorm(y_high_up)
        # y_high_up = self.y_conv1(y_high_up)
        #
        # plus_fuse = x_low + y_high_up
        # plus_fuse = self.plus_conv(plus_fuse)
        # fuse = self.out_conv(torch.cat((plus_fuse,cat_fuse),dim=1))
        # fuse = self.sa(fuse)+fuse


        return fuse

class EMA(nn.Module):
    def __init__(self, channels, rate=1, factor=4):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(int((rate) * channels // self.groups), channels // self.groups, kernel_size=1,
                                 stride=1, padding=0)
        self.conv1x12 = nn.Conv2d(int(channels // self.groups), channels // self.groups, kernel_size=1, stride=1,
                                  padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1,
                                 padding=1, groups=channels // self.groups)

    def forward(self, x1, x2, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        group_x1 = x1.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        group_x2 = x2.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.conv1x1(self.pool_h(group_x1))
        x_w = self.conv1x12(self.pool_w(group_x2)).permute(0, 1, 3, 2)
        hw = torch.cat([x_h, x_w], dim=2)
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()).reshape(b, c, h, w)
        return x1
        # x2 = self.conv3x3(group_x)
        # x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        # return (group_x * weights.sigmoid()).reshape(b, c, h, w)