import torch
import torch.nn as nn
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite
from functools import partial
import pywt
import pywt.data
from timm.layers import DropPath
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .model.lib_mamba.vmambanew import SS2D

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class simam_module(nn.Module):
    def __init__(self, channels, e_lambda=1e-4):
        super(simam_module, self).__init__()

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
        b, c, h, w = x.shape

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(axis=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(axis=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


import torch
import torch.nn as nn


class Large_Selective_Kernel_Network(nn.Module):
    """
    大型选择性卷积核网络（LSKNet）模块

    该模块通过多尺度特征提取和注意力机制增强特征表达能力

    参数：
        dim: 输入特征的通道维度
    """

    def __init__(self, dim):
        super().__init__()

        # 第一路径：5x5深度可分离卷积
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # 第二路径：带膨胀的7x7深度可分离卷积
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3
        )

        # 特征降维卷积（1x1卷积）
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)  # 处理conv0输出
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)  # 处理conv_spatial输出

        # 注意力权重生成层
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)

        # 特征升维卷积
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        """
        前向传播

        参数：
            x: 输入张量，形状为 [B, C, H, W]

        返回：
            torch.Tensor: 增强后的特征，形状为 [B, C, H, W]
        """
        # ========== 双路径特征提取 ==========
        # 路径1：5x5卷积特征
        attn1 = self.conv0(x)
        attn1 = self.conv1(attn1)  # [B, C//2, H, W]

        # 路径2：带膨胀的7x7卷积特征
        attn2 = self.conv_spatial(self.conv0(x))
        attn2 = self.conv2(attn2)  # [B, C//2, H, W]

        # ========== 特征聚合 ==========
        # 拼接双路径特征
        attn_cat = torch.cat([attn1, attn2], dim=1)  # [B, C, H, W]

        # 计算平均池化和最大池化特征
        avg_attn = torch.mean(attn_cat, dim=1, keepdim=True)  # [B, 1, H, W]
        max_attn, _ = torch.max(attn_cat, dim=1, keepdim=True)  # [B, 1, H, W]

        # 拼接池化特征
        agg = torch.cat([avg_attn, max_attn], dim=1)  # [B, 2, H, W]

        # ========== 注意力权重生成 ==========
        # 生成空间注意力权重
        sig = self.conv_squeeze(agg).sigmoid()  # [B, 2, H, W]

        # ========== 特征加权融合 ==========
        # 应用注意力权重
        attn_weighted = (
                attn1 * sig[:, 0, :, :].unsqueeze(1) +
                attn2 * sig[:, 1, :, :].unsqueeze(1)
        )  # [B, C//2, H, W]

        # 恢复通道维度
        attn_weighted = self.conv(attn_weighted)  # [B, C, H, W]

        # ========== 特征增强 ==========
        # 与原始输入相乘（残差连接式增强）
        output = x * attn_weighted  # [B, C, H, W]

        return output



class DRFD(nn.Module):
    """
    Dual Resolution Feature Downsampling module
    基于特征统计的自适应衰减高斯模糊
    """

    def __init__(self, dim, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, kernel_size=3):
        super().__init__()
        self.dim = dim
        self.out_dim = dim
        self.kernel_size = kernel_size

        # 高斯核参数范围
        self.max_sigma = 0.5
        self.min_sigma = 0.1

        # 使用nn.Conv2d作为基础高斯卷积（简化实现）
        self.gaussian_conv = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=dim,
            bias=False
        )

        # 初始化高斯核（使用中等sigma）
        base_sigma = (self.max_sigma + self.min_sigma) / 2
        self._update_gaussian_kernel(base_sigma)

        # 特征统计网络（分析特征复杂度）
        self.statistics_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // 8, 4), 1),
            nn.ReLU(),
            nn.Conv2d(max(dim // 8, 4), 3, 1)  # 输出3个控制参数
        )

        # 归一化和激活
        self.norm = norm_layer(dim)
        self.act = act_layer()

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 计算特征统计量
        # 梯度强度（边缘信息）
        if H > 1 and W > 1:
            grad_x = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
            grad_y = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
            edge_intensity = (grad_x.mean() + grad_y.mean()) * 10
        else:
            edge_intensity = torch.tensor(0.1, device=x.device)

        # 通道间方差
        channel_variance = torch.var(x, dim=[2, 3], unbiased=False).mean()

        # 激活稀疏性
        activation_sparsity = (torch.abs(x) < 0.1).float().mean()

        # 2. 特征分析网络输出控制参数
        stats = self.statistics_net(x)  # [B, 3, 1, 1]
        sigma_scale = torch.sigmoid(stats[:, 0:1])  # sigma缩放因子
        blend_ratio = torch.sigmoid(stats[:, 1:2])  # 混合比例
        residual_weight = torch.sigmoid(stats[:, 2:3])  # 残差权重

        # 3. 基于特征统计计算自适应sigma
        # 特征越复杂（边缘强、方差大），sigma越小（减少模糊）
        complexity = edge_intensity * channel_variance
        adaptive_sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * \
                         torch.sigmoid(-complexity.unsqueeze(0).unsqueeze(0) * 2)

        # 结合网络输出调整sigma
        final_sigma = adaptive_sigma * sigma_scale
        final_sigma = torch.clamp(final_sigma, self.min_sigma, self.max_sigma)

        # 4. 更新高斯核权重（为当前batch）
        # 使用batch的平均sigma
        batch_sigma = final_sigma.mean().item()
        self._update_gaussian_kernel(batch_sigma)

        # 5. 应用高斯模糊
        gaussian_out = self.gaussian_conv(x)

        # 6. 自适应混合（根据特征质量调整）
        # 特征越稀疏，越需要保留原始信息
        blend_weight = blend_ratio * (1 - activation_sparsity)
        blended = blend_weight * gaussian_out + (1 - blend_weight) * x

        # 7. 自适应残差连接
        output = self.norm(x + residual_weight * self.act(blended))

        return output

    def _create_gaussian_kernel(self, size, sigma):
        """创建高斯核"""
        kernel_range = torch.arange(-size // 2 + 1, size // 2 + 1,
                                    dtype=torch.float32, device=self.gaussian_conv.weight.device)
        x, y = torch.meshgrid(kernel_range, kernel_range, indexing='ij')

        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2 + 1e-6))
        kernel = kernel / kernel.sum()

        return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, size, size]

    def _update_gaussian_kernel(self, sigma):
        """更新高斯卷积核的权重"""
        kernel = self._create_gaussian_kernel(self.kernel_size, sigma)
        # 为每个通道复制相同的核
        self.gaussian_conv.weight.data = kernel.repeat(self.dim, 1, 1, 1)


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

class DWConv2d_BN_ReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, bn_weight_init=1):
        super().__init__()
        self.add_module('dwconv3x3',
                        nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=in_channels,
                                  bias=False))
        self.add_module('bn1', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=False))
        self.add_module('dwconv1x1',
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=in_channels,
                                  bias=False))
        self.add_module('bn2', nn.BatchNorm2d(out_channels))

        # Initialize batch norm weights
        nn.init.constant_(self.bn1.weight, bn_weight_init)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, bn_weight_init)
        nn.init.constant_(self.bn2.bias, 0)

    @torch.no_grad()
    def fuse(self):
        # Fuse dwconv3x3 and bn1
        dwconv3x3, bn1, relu, dwconv1x1, bn2 = self._modules.values()

        w1 = bn1.weight / (bn1.running_var + bn1.eps) ** 0.5
        w1 = dwconv3x3.weight * w1[:, None, None, None]
        b1 = bn1.bias - bn1.running_mean * bn1.weight / (bn1.running_var + bn1.eps) ** 0.5

        fused_dwconv3x3 = nn.Conv2d(w1.size(1) * dwconv3x3.groups, w1.size(0), w1.shape[2:], stride=dwconv3x3.stride,
                                    padding=dwconv3x3.padding, dilation=dwconv3x3.dilation, groups=dwconv3x3.groups,
                                    device=dwconv3x3.weight.device)
        fused_dwconv3x3.weight.data.copy_(w1)
        fused_dwconv3x3.bias.data.copy_(b1)

        # Fuse dwconv1x1 and bn2
        w2 = bn2.weight / (bn2.running_var + bn2.eps) ** 0.5
        w2 = dwconv1x1.weight * w2[:, None, None, None]
        b2 = bn2.bias - bn2.running_mean * bn2.weight / (bn2.running_var + bn2.eps) ** 0.5

        fused_dwconv1x1 = nn.Conv2d(w2.size(1) * dwconv1x1.groups, w2.size(0), w2.shape[2:], stride=dwconv1x1.stride,
                                    padding=dwconv1x1.padding, dilation=dwconv1x1.dilation, groups=dwconv1x1.groups,
                                    device=dwconv1x1.weight.device)
        fused_dwconv1x1.weight.data.copy_(w2)
        fused_dwconv1x1.bias.data.copy_(b2)

        # Create a new sequential model with fused layers
        fused_model = nn.Sequential(fused_dwconv3x3, relu, fused_dwconv1x1)
        return fused_model

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, )
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim,)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0,)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h)
        self.act = torch.nn.ReLU(inplace=False)
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


def nearest_multiple_of_16(n):
    if n % 16 == 0:
        return n
    else:
        lower_multiple = (n // 16) * 16
        upper_multiple = lower_multiple + 16

        if (n - lower_multiple) < (upper_multiple - n):
            return lower_multiple
        else:
            return upper_multiple




class DRFD_SameSize_Fixed(nn.Module):
    """修复后的DRFD"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # 主路径：Conv -> BN -> ReLU
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_bn = nn.BatchNorm2d(dim)
        self.conv_act = nn.ReLU(inplace=False)

        # 高斯滤波辅助路径
        gauss_kernel = self._create_gaussian_kernel(5, 0.2)
        self.gaussian = nn.Conv2d(dim, dim, kernel_size=5, stride=1,
                                  padding=2, groups=dim, bias=False)
        self.gaussian.weight.data = gauss_kernel.repeat(dim, 1, 1, 1)
        self.gaussian.weight.requires_grad = False

        self.gauss_bn = nn.BatchNorm2d(dim)
        self.gauss_act = nn.ReLU(inplace=False)
        self.simam = simam_module(dim)
        # 最大池化路径
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=False)
        )

    def _create_gaussian_kernel(self, size: int, sigma: float):
        """创建高斯核"""
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) * math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-size // 2 + 1, size // 2 + 1)]
            for y in range(-size // 2 + 1, size // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()

    def forward(self, x):
        """
        输入: [B, C, H, W]
        输出: [B, C, H, W]
        """
        identity = x

        # 主路径：卷积
        x_conv = self.conv(x)
        x_conv = self.conv_act(self.conv_bn(x_conv))

        # 高斯辅助路径
        gaussian_feat = self.gaussian(x_conv)
        gaussian_feat = self.gauss_act(self.gauss_bn(gaussian_feat))

        # 特征增强
        enhanced = gaussian_feat
        #
        # # 并行处理
        # max_feat = self.max_pool(enhanced)
        # conv_feat = enhanced  # 保持原尺寸
        #
        # # 特征融合
        # combined = torch.cat([conv_feat, max_feat], dim=1)
        # fused = self.fusion(combined)
        #
        # # 残差连接
        # output = identity + fused

        return enhanced



class WaveletTransform(nn.Module):
    """小波变换模块（前向变换）"""

    def __init__(self, in_channels, wt_type='haar'):
        super(WaveletTransform, self).__init__()

        self.in_channels = in_channels
        self.wt_filter, _ = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)

        # self.mask = SSPCAB(3*in_channels)
    def forward(self, x):
        """执行小波变换

        Args:
            x: 输入张量 [B, C, H, W]

        Returns:
            wavelet_coeff: 小波系数 [B, C, 4, H//2, W//2]
        """
        # 确保尺寸为偶数
        if (x.shape[2] % 2 > 0) or (x.shape[3] % 2 > 0):
            pads = (0, x.shape[3] % 2, 0, x.shape[2] % 2)
            x = F.pad(x, pads)

        # 执行小波变换
        wavelet_coeff = self.wt_function(x)

        return wavelet_coeff

class MBWTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1',ssm_ratio=1,forward_type="v05",):
        super(MBWTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels

        self.global_atten =SS2D(d_model=in_channels, d_state=1,
             ssm_ratio=ssm_ratio, initialize="v2", forward_type=forward_type, channel_first=True, k_group=2)



    def forward(self, x):

        x = self.global_atten(x)

        return x

class MobileMambaModule(torch.nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25,
                 kernels=3, ssm_ratio=1, forward_type="v052d",):
        super().__init__()
        self.dim = dim
        self.global_channels = int(global_ratio * dim)
        if self.global_channels + int(local_ratio * dim) > dim:
            self.local_channels = dim - self.global_channels
        else:
            self.local_channels = int(local_ratio * dim)
        self.identity_channels = self.dim - self.global_channels - self.local_channels
        if self.local_channels != 0:
            self.local_op = DWConv2d_BN_ReLU(self.local_channels, self.local_channels, kernels)
        else:
            self.local_op = nn.Identity()
        if self.global_channels != 0:
            self.global_op = MBWTConv2d(self.global_channels, self.global_channels, kernels, wt_levels=1, ssm_ratio=ssm_ratio, forward_type=forward_type,)
        else:
            self.global_op = nn.Identity()

        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            dim, dim, bn_weight_init=0,))

    def forward(self, x):  # x (B,C,H,W)
        x1, x2, x3 = torch.split(x, [self.global_channels, self.local_channels, self.identity_channels], dim=1)
        x1 = self.global_op(x1)
        x2 = self.local_op(x2)
        x = self.proj(torch.cat([x1, x2, x3], dim=1))
        return x


class MobileMambaBlockWindow(torch.nn.Module):
    def __init__(self, dim, global_ratio=0.25, local_ratio=0.25,
                 kernels=5, ssm_ratio=1, forward_type="v052d",):
        super().__init__()
        self.dim = dim
        self.attn = MobileMambaModule(dim, global_ratio=global_ratio, local_ratio=local_ratio,
                                           kernels=kernels, ssm_ratio=ssm_ratio, forward_type=forward_type,)

    def forward(self, x):
        x = self.attn(x)
        return x


class MobileMambaBlock(torch.nn.Module):
    def __init__(self, type,
                 ed, global_ratio=0.25, local_ratio=0.25,
                 kernels=5,  drop_path=0., has_skip=True, ssm_ratio=1, forward_type="v052d"):
        super().__init__()

        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.))
        self.ffn0 = Residual(FFN(ed, int(ed * 2)))

        if type == 's':
            self.mixer = Residual(MobileMambaBlockWindow(ed, global_ratio=global_ratio, local_ratio=local_ratio,
                                                       kernels=kernels, ssm_ratio=ssm_ratio,forward_type=forward_type))

        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.,))
        self.ffn1 = Residual(FFN(ed, int(ed * 2)))

        self.has_skip = has_skip
        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))
        x = (shortcut + self.drop_path(x)) if self.has_skip else x
        return x

from.fusion import CAFMAttention




class EMA(nn.Module):
    def __init__(self, channels, rate=1,factor=4):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        # self.pool_h = nn.AdaptiveAvgPool2d((1,None))
        # self.pool_w = nn.AdaptiveAvgPool2d((None,1))
        self.pool_h = nn.AdaptiveAvgPool2d((None,1))
        self.pool_w = nn.AdaptiveAvgPool2d((1,None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        # self.conv1x1 = nn.Conv2d(int((rate)*channels // self.groups), channels // self.groups, kernel_size=1, stride=1, padding=0)
        # self.conv1x12 = nn.Conv2d(int(channels // self.groups), channels // self.groups, kernel_size=1, stride=1,
        #                          padding=0)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0,groups=channels // self.groups)

    def forward(self, x1,x2,x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        group_x1 = x1.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        group_x2 = x2.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x1)
        x_w = self.pool_w(group_x2).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()).reshape(b, c, h, w)
        return x1

class MSHF(nn.Module):
    def __init__(self, n_channels, kernel=3):
        super(MSHF, self).__init__()

        pad = int((kernel - 1) / 2)

        self.grad_xx = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1,
                                 dilation=1, groups=n_channels, bias=True)
        self.grad_yy = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1,
                                 dilation=1, groups=n_channels, bias=True)
        self.grad_xy = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1,
                                 dilation=1, groups=n_channels, bias=True)

        for m in self.modules():
            if m == self.grad_xx:
                m.weight.data.zero_()
                m.weight.data[:, :, 1, 0] = 1
                m.weight.data[:, :, 1, 1] = -2
                m.weight.data[:, :, 1, -1] = 1
            elif m == self.grad_yy:
                m.weight.data.zero_()
                m.weight.data[:, :, 0, 1] = 1
                m.weight.data[:, :, 1, 1] = -2
                m.weight.data[:, :, -1, 1] = 1
            elif m == self.grad_xy:
                m.weight.data.zero_()
                m.weight.data[:, :, 0, 0] = 1
                m.weight.data[:, :, 0, -1] = -1
                m.weight.data[:, :, -1, 0] = -1
                m.weight.data[:, :, -1, -1] = 1

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False

    def forward(self, x):
        fxx = self.grad_xx(x)
        fyy = self.grad_yy(x)
        fxy = self.grad_xy(x)
        hessian = ((fxx + fyy) + ((fxx - fyy) ** 2 + 4 * (fxy ** 2)) ** 0.5) / 2
        return hessian



class CAA_HSFPN(nn.Module):
    def __init__(self, ch, flag=True, h_kernel_size=3, v_kernel_size=3) -> None:
        super(CAA_HSFPN, self).__init__()

        self.h_conv = nn.Conv2d(ch, ch//2, (1, 3), 1, (0, 3//2), 1, ch//2)
        self.v_conv = nn.Conv2d(ch, ch//2, (3, 1), 1, (3//2, 0), 1, ch//2)
        self.attn = EMA(ch)

        self.m1 = MobileMambaBlock(type='s',
                                    ed=ch//2,  # 嵌入维度
                                    global_ratio=1,
                                    local_ratio=0,
                                    kernels=3,
                                    drop_path=0.1,
                                    has_skip=True,
                                    ssm_ratio=1,
                                    forward_type="v051d")

        self.m2 = MobileMambaBlock(type='s',
                                   ed=ch//2,  # 嵌入维度
                                   global_ratio=1,
                                   local_ratio=0,
                                   kernels=3,
                                   drop_path=0.1,
                                   has_skip=True,
                                   ssm_ratio=1,
                                   forward_type="v051d")

        self.conv = Conv2d_BN(ch,ch,1)

    def forward(self, x):
        y3_1 = self.h_conv(x)
        y3_2 = self.v_conv(x)
        y4 = self.m1(y3_1)
        y5 = self.m2(y3_2)
        y6 = self.conv(torch.cat((y4,y5),dim=1))
        y = self.attn(y6,y6,y6)+x

        return y





class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w

        z = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        z = torch.nn.Parameter(z, requires_grad=True)
        self.z = z


        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor1 = self.mix_block(self.w)
        mix_factor2 = self.mix_block(self.z)
        out1 = fea1 * mix_factor1.expand_as(fea1)
        out2 = fea2 * mix_factor2.expand_as(fea2)
        return out1, out2



# class AFGCAttention(nn.Module):
#     # https://www.sciencedirect.com/science/article/abs/pii/S0893608024002387
#     # https://github.com/Lose-Code/UBRFC-Net
#     # Adaptive Fine-Grained Channel Attention
#     def __init__(self, channel, b=1, gamma=2):
#         super(AFGCAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)#全局平均池化
#         #一维卷积
#         t = int(abs((math.log(channel, 2) + b) / gamma))
#         k = t if t % 2 else t + 1
#         self.conv1 = nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)
#         self.conv2 = nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)
#         self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
#         self.sigmoid = nn.Sigmoid()
#         self.mix = Mix()
#
#     def forward(self, input1,input2):
#         #x1 = self.avg_pool(input1)
#         x2 = self.avg_pool(input1)
#         x1 = self.conv1(x2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)#(1,64,1)
#         x2 = self.fc(x2).squeeze(-1).transpose(-1, -2)#(1,1,64)
#         out1 = torch.sum(torch.matmul(x1,x2),dim=1).unsqueeze(-1).unsqueeze(-1)#(1,64,1,1)
#         out1 = self.sigmoid(out1)
#         out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),x1.transpose(-1, -2)),dim=1).unsqueeze(-1).unsqueeze(-1)
#         out2 = self.sigmoid(out2)
#         out1, out2 = self.mix(out1,out2)
#         out1 = self.conv2(out1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#
#         out2 = self.conv2(out2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         out2 = (out1+out2)*input1
#
#         return input2*out2



class AFGCAttention(nn.Module):
    # https://www.sciencedirect.com/science/article/abs/pii/S0893608024002387
    # https://github.com/Lose-Code/UBRFC-Net
    # Adaptive Fine-Grained Channel Attention
    def __init__(self, channel, b=1, gamma=2):
        super(AFGCAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#全局平均池化
        #一维卷积
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, padding=0, bias=False)
        self.fc = nn.Conv2d(2*channel, 2*channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()

    def forward(self, input1,input2):
        #x1 = self.avg_pool(input1)
        input = torch.cat([input1,input2],dim=1)
        x2 = self.avg_pool(input)
        x1 = self.conv1(x2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)#(1,64,1)
        x2 = self.fc(x2).squeeze(-1).transpose(-1, -2)#(1,1,64)
        out1 = torch.sum(torch.matmul(x1,x2),dim=1).unsqueeze(-1).unsqueeze(-1)#(1,64,1,1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),x1.transpose(-1, -2)),dim=1).unsqueeze(-1).unsqueeze(-1)
        out2 = self.sigmoid(out2)
        out1, out2 = self.mix(out1,out2)
        out1 = self.conv2(out1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out2 = self.conv2(out2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out2 = out1+out2
        w1, w2 = torch.split(out2, out2.size(1) // 2, dim=1)
        input1 = input1*w1
        input2 = input2*w2

        return input1*input2



class WaveletDomainProcessing(nn.Module):
    """小波域特征提取模块"""

    def __init__(self, in_channels, hf=False,wt_levels=1):
        super(WaveletDomainProcessing, self).__init__()
        self.hf = hf
        self.wt_levels = wt_levels
        self.in_channels = in_channels
        self.llattn = CAA_HSFPN(in_channels)
        self.hhattn2 = AFGCAttention(in_channels)
        self.ffn = DRFD_SameSize_Fixed(in_channels)


    def forward(self, wavelet_coeff, level):
        """在小波域进行特征提取

        Args:
            wavelet_coeff: 小波系数 [B, C, 4, H, W]
            level: 当前处理的小波分解层级

        Returns:
            processed_coeff: 处理后的小波系数 [B, C, 4, H, W]
        """
        # 重塑为2D卷积格式
        shape_orig = wavelet_coeff.shape
        ll_coeff = wavelet_coeff[:, :, 0, :, :].reshape(shape_orig[0], shape_orig[1], shape_orig[3], shape_orig[4])  # 低频分量
        h1_coeff = wavelet_coeff[:, :, 1, :, :].reshape(shape_orig[0], shape_orig[1], shape_orig[3], shape_orig[4])  # 低频分量
        h2_coeff = wavelet_coeff[:, :, 2, :, :].reshape(shape_orig[0], shape_orig[1], shape_orig[3], shape_orig[4])  # 低频分量
        h3_coeff = wavelet_coeff[:, :, 3, :, :].reshape(shape_orig[0], shape_orig[1], shape_orig[3], shape_orig[4])  # 低频分量


        # if self.hf: h_coeff = self.hhattn2(self.ffn(h1_coeff+h2_coeff+h3_coeff)) processed_coeff = torch.stack([
        # ll_coeff, h1_coeff*h_coeff+h1_coeff, h2_coeff*h_coeff+h2_coeff, h3_coeff*h_coeff+h3_coeff], dim=2)
        if self.hf:
            h_coeff = self.ffn(h1_coeff+h2_coeff+h3_coeff)
            h1_coeff = self.hhattn2(h_coeff,h1_coeff)+h1_coeff
            h2_coeff = self.hhattn2(h_coeff,h2_coeff)+h2_coeff
            h3_coeff = self.hhattn2(h_coeff,h3_coeff)+h3_coeff

            processed_coeff = torch.stack([ll_coeff, h1_coeff, h2_coeff, h3_coeff], dim=2)
        else:
            ll_coeff = self.llattn(ll_coeff)
            processed_coeff = torch.stack([ll_coeff, h1_coeff, h2_coeff, h3_coeff], dim=2)

        return processed_coeff

    def process_multilevel(self, wavelet_coeff_list):
        """处理多级小波分解系数

        Args:
            wavelet_coeff_list: 各级小波系数列表

        Returns:
            ll_coeff_list: 低频系数列表
            h_coeff_list: 高频系数列表
        """
        ll_coeff_list = []
        h_coeff_list = []

        for level, coeff in enumerate(wavelet_coeff_list):
            processed_coeff = self.forward(coeff, level)

            # 分离低频和高频分量
            ll_coeff = processed_coeff[:, :, 0, :, :]  # 低频分量
            h_coeff = processed_coeff[:, :, 1:4, :, :]  # 高频分量

            ll_coeff_list.append(ll_coeff)
            h_coeff_list.append(h_coeff)

        return ll_coeff_list, h_coeff_list



class MultilevelWaveletReconstruction(nn.Module):
    """多级小波重构模块"""

    def __init__(self, in_channels, wt_type='haar'):
        super(MultilevelWaveletReconstruction, self).__init__()

        self.inverse_transform = InverseWaveletTransform(in_channels, wt_type)

    def forward(self, ll_coeff_list, h_coeff_list, original_shapes):
        """执行多级小波重构

        Args:
            ll_coeff_list: 低频系数列表（从粗到细）
            h_coeff_list: 高频系数列表（从粗到细）
            original_shapes: 各级原始形状列表

        Returns:
            final_reconstructed: 最终重构图像
        """
        next_ll_coeff = 0

        # 从最深层开始逐级重构
        for level in range(len(ll_coeff_list) - 1, -1, -1):
            curr_ll = ll_coeff_list[level]
            curr_h = h_coeff_list[level]
            curr_shape = original_shapes[level]

            # 合并当前层的低频和上层重构结果
            curr_ll = curr_ll + next_ll_coeff

            # 执行小波反变换
            next_ll_coeff = self.inverse_transform(curr_ll, curr_h, curr_shape[2:])

        return next_ll_coeff


class InverseWaveletTransform(nn.Module):
    """小波反变换模块"""

    def __init__(self, in_channels, wt_type='haar'):
        super(InverseWaveletTransform, self).__init__()

        self.in_channels = in_channels
        _, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

    def forward(self, ll_coeff, h_coeff, target_shape=None):
        """执行小波反变换

        Args:
            ll_coeff: 低频系数 [B, C, H, W]
            h_coeff: 高频系数 [B, C, 3, H, W]
            target_shape: 目标输出形状 (H, W)，用于裁剪

        Returns:
            reconstructed: 重建后的图像 [B, C, H, W]
        """
        # 合并低频和高频分量
        wavelet_coeff = torch.cat([ll_coeff.unsqueeze(2), h_coeff], dim=2)

        # 执行小波反变换
        reconstructed = self.iwt_function(wavelet_coeff)

        # 裁剪到目标形状
        if target_shape is not None:
            reconstructed = reconstructed[:, :, :target_shape[0], :target_shape[1]]

        return reconstructed

from einops import rearrange  # 用于灵活调整张量维度的工具库
class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, bias=True):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C / 2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 2 / 3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 3 / 4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 4 / 5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



# 使用示例
class WaveletProcessingPipeline(nn.Module):
    """完整的小波处理流程"""

    def __init__(self, in_channels,hf=False,wt_levels=1, wt_type='haar'):
        super(WaveletProcessingPipeline, self).__init__()

        self.wt_levels = wt_levels

        # 初始化各模块
        self.wavelet_transform = WaveletTransform(in_channels, wt_type)
        self.wavelet_domain = WaveletDomainProcessing(in_channels,hf,wt_levels)
        self.wavelet_reconstruction = MultilevelWaveletReconstruction(in_channels, wt_type)


    def forward(self, x):
        # 存储各级原始形状
        original_shapes = []
        wavelet_coeff_list = []

        # 多级小波分解
        curr_x = x
        for level in range(self.wt_levels):
            original_shapes.append(curr_x.shape)
            coeff = self.wavelet_transform(curr_x)
            wavelet_coeff_list.append(coeff)

            # 提取下一级的低频分量
            curr_x = coeff[:, :, 0, :, :]

        ll_coeff_list, h_coeff_list = self.wavelet_domain.process_multilevel(wavelet_coeff_list)


        # 多级小波重构
        reconstructed = self.wavelet_reconstruction(ll_coeff_list, h_coeff_list, original_shapes)+x



        return reconstructed

        # ll_coeff_list, h_coeff_list = self.wavelet_domain.process_multilevel(wavelet_coeff_list)
        #
        # # 多级小波重构
        # reconstructed = self.wavelet_reconstruction(ll_coeff_list, h_coeff_list, original_shapes)
        #
        # return reconstructed




class ACDFF(nn.Module):
    """完整的小波处理流程"""

    def __init__(self, in_channels,out_inchannels, wt_type='haar'):
        super(ACDFF, self).__init__()



        self.wavelet_domain = Conv2d_BN(in_channels, out_inchannels, 1)
        self.pixel = WaveletProcessingPipeline(out_inchannels)

    def forward(self, x):

        up = self.wavelet_domain(x)
        out = self.pixel(up)

        return out


class multiFFN(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(multiFFN, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2
        # PW first or DW first?
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim*2, 1),
        )

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1,
                      groups=self.dim_sp),
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=5, padding=2,
                      groups=self.dim_sp),
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=7, padding=3,
                      groups=self.dim_sp),
        )

        self.gelu = nn.GELU()
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
        )


    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim_sp, dim=1))
        x[1] = self.conv1_1(x[1])
        x[2] = self.conv1_2(x[2])
        x[3] = self.conv1_3(x[3])
        x = torch.cat(x, dim=1)
        x = self.gelu(x)
        x = self.conv_fina(x)


        return x

class channel_att(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(channel_att, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1)
        y = y.transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class local_att(nn.Module):
    def __init__(self, channel, reduction=4):
        super(local_att, self).__init__()

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        _, _, h, w = x.size()

        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))

        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)

        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out

#Channel and Position Attention Mechanism (CPAM)
class CPAM(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.channel_att = channel_att(ch)
        self.local_att = local_att(ch)

    def forward(self, x):
        input1=x
        input1 = self.channel_att(input1)
        x = input1
        x = self.local_att(x)
        return x


from .MSAA import SegNext_Attention
from .cswin_transformer import CSWinBlock
from .fusion import Freq_Fusion
class BCDFF(nn.Module):
    """完整的小波处理流程"""

    def __init__(self, in_channels,out_channels):
        super(BCDFF, self).__init__()

        self.multilevel2 = Freq_Fusion()

    def forward(self, x):

        xh,xl = x


        out = self.multilevel2(xl,xh)

        return out






