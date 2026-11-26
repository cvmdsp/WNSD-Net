import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
# from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
# train speed is slower after enabling this opts.
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

try:
    from .csm_triton import cross_scan_fn, cross_merge_fn
except:
    from csm_triton import cross_scan_fn, cross_merge_fn

try:
    from .csm_tritonk2 import cross_scan_fn_k2, cross_merge_fn_k2
    from .csm_tritonk2 import cross_scan_fn_k2_torch, cross_merge_fn_k2_torch
except:
    from csm_tritonk2 import cross_scan_fn_k2, cross_merge_fn_k2
    from csm_tritonk2 import cross_scan_fn_k2_torch, cross_merge_fn_k2_torch

try:
    from .csms6s import selective_scan_fn, selective_scan_flop_jit
except:
    from csms6s import selective_scan_fn, selective_scan_flop_jit

# FLOPs counter not prepared fro mamba2
# try:
#     from .mamba2.ssd_minimal import selective_scan_chunk_fn
# except:
#     from mamba2.ssd_minimal import selective_scan_chunk_fn


# =====================================================
# we have this class as linear and conv init differ from each other
# this function enable loading from both conv2d or linear
class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._patch_merging_pad = self._patch_merging_pad_channel_first if channel_first else self._patch_merging_pad_channel_last
        self.reduction = Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad_channel_last(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    @staticmethod
    def _patch_merging_pad_channel_first(x: torch.Tensor):
        H, W = x.shape[-2:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]  # ... H/2 W/2
        x1 = x[..., 1::2, 0::2]  # ... H/2 W/2
        x2 = x[..., 0::2, 1::2]  # ... H/2 W/2
        x3 = x[..., 1::2, 1::2]  # ... H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class gMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        x = self.fc2(x * self.act(z))
        x = self.drop(x)
        return x


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1).contiguous()).view(B, C, H, W).contiguous()
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C).contiguous()).view(B, H, W, C).contiguous()
        else:
            raise NotImplementedError


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
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


# =====================================================
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))  # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))  # (K, inner)
        del dt_projs

        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


class SS2Dv2:
    def __initv2__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU(inplace=False),
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v05",
            channel_first=False,
            # ======================
            k_group=1,
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(forward_type, d_inner, channel_first)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba",
                        scan_force_torch=True),
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="mamba"),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="oflex"),
            v04=partial(self.forward_corev2, force_fp32=False),  # selective_scan_backend="oflex", scan_mode="cross2d"
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),
            # selective_scan_backend="oflex", scan_mode="cross2d"
            # ===============================
            v051d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"),
            v052d=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"),
            v052dc=partial(self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="cascade2d"),
            # ===============================
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), selective_scan_backend="core"),
            v3=partial(self.forward_corev2, force_fp32=False, selective_scan_backend="oflex"),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        self.k_group = k_group

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Conv2d_BN(d_model, d_proj)
        # self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act = nn.SiLU(inplace=False)

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            # torch.nn.Conv2d(d_inner, (dt_rank + d_state * 2), 1, bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Conv2d_BN(d_inner, d_model)
        # self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = mamba_init.init_dt_A_D(
                d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=k_group,
            )
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn((k_group, d_inner, dt_rank)))  # 0.1 is added in 0430
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((k_group, d_inner)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(
                torch.zeros((k_group * d_inner, d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))


    def _init_attention_modules(self, **kwargs):
        """初始化注意力相关模块"""
        d_model = kwargs.get('d_model', 96)
        k_group = kwargs.get('k_group', 4)

        if self.attention_type == "simam":
            self.simam = SIMAM(d_model)
        elif self.attention_type == "learnable":
            self.attention_scan = EfficientAttentionScan(d_model, num_directions=k_group)

        # 保存原始的forward_core函数
        self._original_forward_core = self.forward_core

    def compute_attention_scan_indices(self, x):
        """计算注意力扫描的索引"""
        B, C, H, W = x.shape

        if self.attention_type == "simam":
            # 使用SIMAM计算注意力
            with torch.no_grad():
                attn_out = self.simam(x)
                attention_map = attn_out.mean(dim=1, keepdim=True)  # (B, 1, H, W)

                # 生成k_group个不同的扫描顺序
                scan_indices_list = []
                for k in range(self.k_group):
                    # 可以添加不同的扰动来生成不同的扫描顺序
                    if k == 0:
                        # 第一个方向：按注意力降序
                        scores = attention_map.view(B, -1)
                    else:
                        # 其他方向：添加噪声扰动
                        noise = torch.randn_like(attention_map) * 0.1
                        scores = (attention_map + noise).view(B, -1)

                    _, indices = torch.sort(scores, dim=1, descending=True)
                    scan_indices_list.append(indices)

                return scan_indices_list

        elif self.attention_type == "learnable":
            # 使用可学习的注意力扫描
            _, scan_indices_list = self.attention_scan(x, return_indices=True)
            return scan_indices_list

        return None

    def apply_attention_scan(self, x, scan_indices_list):
        """应用注意力扫描"""
        B, C, H, W = x.shape
        L = H * W
        K = len(scan_indices_list)

        # 准备结果张量
        x_scanned_all = torch.zeros(B, K, C, L, device=x.device, dtype=x.dtype)

        # 对每个扫描方向应用重排
        for k in range(K):
            scan_indices = scan_indices_list[k]
            x_flat = x.view(B, C, L)
            batch_idx = torch.arange(B).view(B, 1).expand(-1, L)
            x_scanned_all[:, k] = x_flat[batch_idx, :, scan_indices]

        return x_scanned_all

    def restore_from_attention_scan(self, y_scanned_all, scan_indices_list):
        """从注意力扫描结果恢复原始顺序"""
        B, K, C, L = y_scanned_all.shape
        H = W = int(L ** 0.5)

        # 合并所有方向的结果
        y_restored_all = torch.zeros(B, K, C, L, device=y_scanned_all.device)

        for k in range(K):
            scan_indices = scan_indices_list[k]
            inverse_indices = torch.argsort(scan_indices, dim=1)
            batch_idx = torch.arange(B).view(B, 1).expand(-1, L)
            y_restored_all[:, k] = y_scanned_all[:, k][batch_idx, :, inverse_indices]

        # 合并所有方向（简单求和）
        y_restored = y_restored_all.sum(dim=1).view(B, C, H, W)

        return y_restored

    def forward_core_with_attention(self, x, **kwargs):
        """支持注意力扫描的forward_core"""
        if not self.use_attention_scan:
            return self._original_forward_core(x, **kwargs)

        B, D, H, W = x.shape
        L = H * W

        # 1. 计算注意力扫描索引
        scan_indices_list = self.compute_attention_scan_indices(x)

        if scan_indices_list is None:
            # 回退到原始实现
            return self._original_forward_core(x, **kwargs)

        # 2. 应用扫描
        outputs_list = []

        for k in range(self.k_group):
            # 获取当前扫描路径的索引
            scan_indices = scan_indices_list[k]  # (B, L)

            # 根据扫描路径重排特征
            x_scanned_k = self.apply_single_scan_path(x, scan_indices)  # (B, D, H, W)

            # 使用原始SS2D进行处理（保持2D结构）
            # 这里调用原始的forward_core，处理重排后的2D特征
            y_scanned_k = self._original_forward_core(x_scanned_k, **kwargs)  # (B, D, H, W)

            # 恢复原始顺序
            y_restored_k = self.restore_single_scan_path(y_scanned_k, scan_indices)

            outputs_list.append(y_restored_k)

        # 3. 合并多个扫描路径的结果
        # 根据注意力权重进行加权融合
        if self.k_group > 1:
            # 计算每个路径的注意力权重
            attn_weights = self.compute_attention_weights(x)  # (B, K, H, W)

            # 加权融合
            y_final = torch.zeros_like(x)
            for k in range(self.k_group):
                weight = attn_weights[:, k:k + 1, :, :]  # (B, 1, H, W)
                y_final += outputs_list[k] * weight
        else:
            y_final = outputs_list[0]

        return y_final

    def apply_single_scan_path(self, x, scan_indices):
        """应用单个扫描路径的重排"""
        B, D, H, W = x.shape
        L = H * W

        # 将2D特征展平为序列
        x_flat = x.reshape(B, D, L)  # (B, D, L)

        # 根据扫描索引重排
        batch_indices = torch.arange(B, device=x.device)[:, None, None]  # (B, 1, 1)
        channel_indices = torch.arange(D, device=x.device)[None, :, None]  # (1, D, 1)

        # 扩展扫描索引以匹配通道维度
        scan_indices_expanded = scan_indices[:, None, :].expand(-1, D, -1)  # (B, D, L)

        # 收集重排后的特征
        x_scanned = torch.gather(
            x_flat,
            dim=2,
            index=scan_indices_expanded
        )  # (B, D, L)

        # 重塑回2D
        x_scanned_2d = x_scanned.reshape(B, D, H, W)

        return x_scanned_2d

    def restore_single_scan_path(self, y_scanned, scan_indices):
        """恢复单个扫描路径的原始顺序"""
        B, D, H, W = y_scanned.shape
        L = H * W

        # 将2D特征展平
        y_scanned_flat = y_scanned.reshape(B, D, L)  # (B, D, L)

        # 创建逆索引（从扫描顺序恢复原始顺序）
        # 我们需要为每个位置找到它在扫描顺序中的位置
        batch_indices = torch.arange(B, device=y_scanned.device)[:, None]  # (B, 1)

        # 创建逆索引映射
        reverse_indices = torch.zeros_like(scan_indices)
        for b in range(B):
            for pos in range(L):
                # 找到原始位置pos在扫描顺序中的索引
                scanned_pos = torch.where(scan_indices[b] == pos)[0]
                if len(scanned_pos) > 0:
                    reverse_indices[b, pos] = scanned_pos[0]

        # 扩展逆索引以匹配通道维度
        reverse_indices_expanded = reverse_indices[:, None, :].expand(-1, D, -1)  # (B, D, L)

        # 根据逆索引恢复原始顺序
        y_restored = torch.gather(
            y_scanned_flat,
            dim=2,
            index=reverse_indices_expanded
        )  # (B, D, L)

        # 重塑回2D
        y_restored_2d = y_restored.reshape(B, D, H, W)

        return y_restored_2d

    def compute_attention_weights(self, x):
        """计算多个扫描路径的注意力权重"""
        B, D, H, W = x.shape

        # 使用轻量级卷积计算注意力
        if not hasattr(self, 'attn_conv'):
            self.attn_conv = nn.Sequential(
                nn.Conv2d(D, D // 4, 3, padding=1, groups=min(D, D // 4)),
                nn.GELU(),
                nn.Conv2d(D // 4, self.k_group, 1),
                nn.Softmax(dim=1)
            ).to(x.device)

        attn_weights = self.attn_conv(x)  # (B, K, H, W)

        return attn_weights


    def _simplified_1d_ssm(self, x):
        """简化的1D状态空间计算（用于演示）"""
        # 在实际应用中，这里应该调用真正的1D SSM
        # 这里使用简单的投影作为演示
        B, L, D = x.shape

        # 简化的状态空间计算
        # 实际应该使用：y = selective_scan(x, ...)
        y = F.linear(x, self._get_ssm_weights(D))

        return y

    def _get_ssm_weights(self, D):
        """获取状态空间权重（简化版本）"""
        # 这里应该返回真正的SSM参数
        # 简化版本：返回单位矩阵
        return torch.eye(D, device=self.dt_projs_weight.device)


    def forward_corev2(
            self,
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32
            # ==============================
            ssoflex=True,  # True: input 16 or 32 output 32 False: output dtype as input
            no_einsum=False,  # replace einsum with linear or conv1d to raise throughput
            # ==============================
            selective_scan_backend=None,
            # ==============================
            scan_mode="cross2d",
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):
        assert scan_mode in ["unidi", "bidi", "cross2d", "cascade2d"]
        assert selective_scan_backend in [None, "oflex", "core", "mamba", "torch"]
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode]

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            # print(u.device)
            # print(selective_scan_backend)
            if u.device == torch.device("cpu"):
                selective_scan_backend = "torch"
            else:
                selective_scan_backend = "oflex"
            return selective_scan_fn(u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex,
                                     backend=selective_scan_backend)

        if _scan_mode == 3:
            x_proj_bias = getattr(self, "x_proj_bias", None)

            def scan_rowcol(
                    x: torch.Tensor,
                    proj_weight: torch.Tensor,
                    proj_bias: torch.Tensor,
                    dt_weight: torch.Tensor,
                    dt_bias: torch.Tensor,  # (2*c)
                    _As: torch.Tensor,  # As = -torch.exp(A_logs.to(torch.float))[:2,] # (2*c, d_state)
                    _Ds: torch.Tensor,
                    width=True,
            ):
                # x: (B, D, H, W)
                # proj_weight: (2 * D, (R+N+N))
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)  # (B, H, 2, D, W)
                if no_einsum:
                    x_dbl = F.conv1d(xs.view(_B, -1, _L), proj_weight.view(-1, _D, 1),
                                     bias=(proj_bias.view(-1) if proj_bias is not None else None), groups=2)
                    dts, Bs, Cs = torch.split(x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2)
                    dts = F.conv1d(dts.contiguous().view(_B, -1, _L), dt_weight.view(2 * _D, -1, 1), groups=2)
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)

                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)

                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)

                ys: torch.Tensor = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys

            As = -self.A_logs.to(torch.float).exp().view(self.k_group, -1, N).contiguous()
            x = F.layer_norm(x.permute(0, 2, 3, 1), normalized_shape=(int(x.shape[1]),)).permute(0, 3, 1,
                                                                                                 2).contiguous()  # added0510 to avoid nan
            y_row = scan_rowcol(
                x,
                proj_weight=self.x_proj_weight.view(self.k_group, -1, D)[:2].contiguous(),
                proj_bias=(x_proj_bias.view(self.k_group, -1)[:2].contiguous() if x_proj_bias is not None else None),
                dt_weight=self.dt_projs_weight.view(self.k_group, D, -1)[:2].contiguous(),
                dt_bias=(self.dt_projs_bias.view(self.k_group, -1)[
                         :2].contiguous() if self.dt_projs_bias is not None else None),
                _As=As[:2].contiguous().view(-1, N),
                _Ds=self.Ds.view(self.k_group, -1)[:2].contiguous().view(-1),
                width=True,
            ).view(B, H, 2, -1, W).sum(dim=2).permute(0, 2, 1, 3).contiguous() # (B,C,H,W)
            y_row = F.layer_norm(y_row.permute(0, 2, 3, 1), normalized_shape=(int(y_row.shape[1]),)).permute(0, 3, 1,
                                                                                                             2).contiguous()  # added0510 to avoid nan
            y_col = scan_rowcol(
                y_row,
                proj_weight=self.x_proj_weight.view(self.k_group, -1, D)[2:].contiguous().to(y_row.dtype),
                proj_bias=(
                    x_proj_bias.view(self.k_group, -1)[2:].contiguous().to(
                        y_row.dtype) if x_proj_bias is not None else None),
                dt_weight=self.dt_projs_weight.view(self.k_group, D, -1)[2:].contiguous().to(y_row.dtype),
                dt_bias=(self.dt_projs_bias.view(self.k_group, -1)[2:].contiguous().to(
                    y_row.dtype) if self.dt_projs_bias is not None else None),
                _As=As[2:].contiguous().view(-1, N),
                _Ds=self.Ds.view(self.k_group, -1)[2:].contiguous().view(-1),
                width=False,
            ).view(B, W, 2, -1, H).sum(dim=2).permute(0, 2, 3, 1).contiguous()
            y = y_col
        else:
            x_proj_bias = getattr(self, "x_proj_bias", None)
            if self.k_group == 4:
                xs = cross_scan_fn(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                   force_torch=scan_force_torch)
            else:
                xs = cross_scan_fn_k2(x, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                      force_torch=scan_force_torch)
            if no_einsum:
                x_dbl = F.conv1d(xs.view(B, -1, L).contiguous(), self.x_proj_weight.view(-1, D, 1).contiguous(),
                                 bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L).contiguous(), [R, N, N], dim=2)
                dts = F.conv1d(dts.contiguous().view(B, -1, L).contiguous(), self.dt_projs_weight.view(K * D, -1, 1).contiguous(), groups=K)
            else:
                x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1).contiguous()
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

            xs = xs.view(B, -1, L).contiguous()
            dts = dts.contiguous().view(B, -1, L).contiguous()
            As = -self.A_logs.to(torch.float).exp()  # (k * c, d_state)
            Ds = self.Ds.to(torch.float)  # (K * c)
            Bs = Bs.contiguous().view(B, K, N, L).contiguous()
            Cs = Cs.contiguous().view(B, K, N, L).contiguous()
            delta_bias = self.dt_projs_bias.view(-1).contiguous().to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys: torch.Tensor = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W).contiguous()

            if self.k_group == 4:
                y: torch.Tensor = cross_merge_fn(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                                 force_torch=scan_force_torch)
            else:
                y: torch.Tensor = cross_merge_fn_k2(ys, in_channel_first=True, out_channel_first=True, scans=_scan_mode,
                                                    force_torch=scan_force_torch)

            if getattr(self, "__DEBUG__", False):
                setattr(self, "__data__", dict(
                    A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                    us=xs, dts=dts, delta_bias=delta_bias,
                    ys=ys, y=y, H=H, W=W,
                ))

        y = y.view(B, -1, H, W).contiguous()
        if not channel_first:
            y = y.view(B, -1, H * W).contiguous().transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1).contiguous()  # (B, L, C)
#        y = out_norm(y)

        return y.to(x.dtype)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
        z = self.act(z.clone())
        x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm

        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_none:
            out_norm = nn.Identity()
        elif out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        else:
            out_norm = LayerNorm(d_inner)

        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag):] == tag
        if ret:
            value = value[:-len(tag)]
        return ret, value


# mamba2 support ================================
class SS2Dm0:
    def __initm0__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,  # now with mamba2, dstate should be bigger...
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.GELU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v2",
            # ======================
            forward_type="m0",
            # ======================
            with_initial_state=False,
            channel_first=False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        assert d_inner % dt_rank == 0
        self.channel_first = channel_first
        Linear = Linear2d if channel_first else nn.Linear
        self.with_dconv = d_conv > 1
        self.forward = self.forwardm0

        # tags for forward_type ==============================
        checkpostfix = SS2Dv2.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = SS2Dv2.get_outnorm(forward_type, d_inner, False)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            m0=partial(self.forward_corem0, force_fp32=False, dstate=d_state),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 4

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        # self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.in_proj = Conv2d_BN(d_model, d_proj)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        # self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.out_proj = Conv2d_BN(d_inner, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
            self.A_logs = nn.Parameter(torch.randn((k_group, dt_rank)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn((k_group, dt_rank)))  # 0.1 is added in 0430
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group, dt_rank, int(d_inner // dt_rank))))
            self.A_logs = nn.Parameter(torch.zeros((k_group, dt_rank)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, dt_rank)))

        # init state ============================
        self.initial_state = None
        if with_initial_state:
            self.initial_state = nn.Parameter(torch.zeros((1, k_group * dt_rank, int(d_inner // dt_rank), d_state)),
                                              requires_grad=False)

    def forward_corem0(
            self,
            x: torch.Tensor = None,
            # ==============================
            force_fp32=False,  # True: input fp32
            chunk_size=64,
            dstate=64,
            # ==============================
            selective_scan_backend='torch',
            scan_mode="cross2d",
            scan_force_torch=False,
            # ==============================
            **kwargs,
    ):
        assert scan_mode in ["unidi", "bidi", "cross2d"]
        assert selective_scan_backend in [None, "triton", "torch"]
        x_proj_bias = getattr(self, "x_proj_bias", None)
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        N = dstate
        B, H, W, RD = x.shape
        K, R = self.A_logs.shape
        K, R, D = self.Ds.shape
        assert RD == R * D
        L = H * W
        KR = K * R
        _scan_mode = dict(cross2d=0, unidi=1, bidi=2, cascade2d=3)[scan_mode]

        initial_state = None
        if self.initial_state is not None:
            assert self.initial_state.shape[-1] == dstate
            initial_state = self.initial_state.detach().repeat(B, 1, 1, 1)
        xs = cross_scan_fn(x.view(B, H, W, RD).contiguous(), in_channel_first=False, out_channel_first=False,
                           scans=_scan_mode, force_torch=scan_force_torch)  # (B, H, W, 4, D)
        x_dbl = torch.einsum("b l k d, k c d -> b l k c", xs, self.x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, -1, K, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=3)
        xs = xs.contiguous().view(B, L, KR, D).contiguous()
        dts = dts.contiguous().view(B, L, KR).contiguous()
        Bs = Bs.contiguous().view(B, L, K, N).contiguous()
        Cs = Cs.contiguous().view(B, L, K, N).contiguous()
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        As = -self.A_logs.to(torch.float).exp().view(KR).contiguous()
        Ds = self.Ds.to(torch.float).view(KR, D).contiguous()
        dt_bias = self.dt_projs_bias.view(KR).contiguous()

        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        ys, final_state = selective_scan_chunk_fn(
            xs, dts, As, Bs, Cs, chunk_size=chunk_size, D=Ds, dt_bias=dt_bias,
            initial_states=initial_state, dt_softplus=True, return_final_states=True,
            backend=selective_scan_backend,
        )
        y: torch.Tensor = cross_merge_fn(ys.contiguous().view(B, H, W, K, RD).contiguous(), in_channel_first=False,
                                         out_channel_first=False, scans=_scan_mode, force_torch=scan_force_torch)

        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=self.A_logs, Bs=Bs, Cs=Cs, Ds=self.Ds,
                us=xs, dts=dts, delta_bias=self.dt_projs_bias,
                initial_state=self.initial_state, final_satte=final_state,
                ys=ys, y=y, H=H, W=W,
            ))
        if self.initial_state is not None:
            self.initial_state = nn.Parameter(final_state.detach().sum(0, keepdim=True), requires_grad=False)

        y = self.out_norm(y.view(B, H, W, -1).contiguous())

        return y.to(x.dtype)

    def forwardm0(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if self.with_dconv:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)

        y = self.forward_core(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


# class SS2D(nn.Module, SS2Dv2, SS2Dm0):
#     def __init__(
#             self,
#             # basic dims ===========
#             d_model=96,
#             d_state=16,
#             ssm_ratio=2.0,
#             dt_rank="auto",
#             act_layer=nn.SiLU(inplace=False),
#             # dwconv ===============
#             d_conv=3,  # < 2 means no conv
#             conv_bias=True,
#             # ======================
#             dropout=0.0,
#             bias=False,
#             # dt init ==============
#             dt_min=0.001,
#             dt_max=0.1,
#             dt_init="random",
#             dt_scale=1.0,
#             dt_init_floor=1e-4,
#             initialize="v0",
#             # ======================
#             forward_type="v5",
#             channel_first=False,
#             # ======================
#             k_group=4,
#             **kwargs,
#     ):
#         super().__init__()
#         kwargs.update(
#             d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
#             act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
#             dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
#             initialize=initialize, forward_type=forward_type, channel_first=channel_first, k_group=k_group,
#         )
#         if forward_type in ["v0", "v0seq"]:
#             self.__initv0__(seq=("seq" in forward_type), **kwargs)
#         elif forward_type.startswith("xv"):
#             self.__initxv__(**kwargs)
#         elif forward_type.startswith("m"):
#             self.__initm0__(**kwargs)
#         else:
#             self.__initv2__(**kwargs)


import torch
import torch.nn as nn
import torch.nn.functional as F


class SIMAM(nn.Module):
    """SIMAM注意力模块 - 兼容原SS2D接口"""

    def __init__(self, channels, e_lambda=1e-4):
        super().__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.shape
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activation(y)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., input_resolution=(64,64)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.input_resolution = input_resolution
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        flops += 2 * H * W * self.in_features * self.hidden_features
        flops += H * W * self.hidden_features

        return flops

class MSHF(nn.Module):
    def __init__(self, n_channels, kernel=3):
        super(MSHF, self).__init__()

        pad = int((kernel - 1) / 2)

        self.grad_xx = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1,
                                 dilation=1, groups=n_channels, bias=True)
        self.grad_yy = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=5, stride=1, padding=2,
                                 dilation=1, groups=n_channels, bias=True)
        self.grad_xy = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=7, stride=1, padding=3,
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

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        #Squeeze操作
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation操作（FC+ReLU+FC+Sigmoid）
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            h_sigmoid()
        )
    def forward(self,x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ShuffleAttn(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=None, group=4, act_layer=nn.GELU,
                 input_resolution=(64, 64)):
        super().__init__()
        self.group = group
        self.input_resolution = input_resolution
        self.in_features = in_features
        self.out_features = out_features

        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, out_features, groups=self.group, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    def channel_rearrange(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, self.group, group_channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)
        return x

    def forward(self, x):
        x = self.channel_shuffle(x)
        x = self.gating(x)
        x = self.channel_rearrange(x)

        return x

from .dsc import DSConv

class EfficientAttentionScan(nn.Module):
    """高效的注意力引导扫描（针对k_group=2优化）"""

    def __init__(self, d_model, num_directions=2):
        super().__init__()
        self.num_directions = num_directions

        # 轻量级的注意力预测网络
        self.attention_predictor = nn.Sequential(
            DSConv(
                in_ch=d_model,
                out_ch=num_directions,
                kernel_size=11,
                extend_scope=1,
                morph=0,
                if_offset=True, ),
            nn.BatchNorm2d(num_directions),
            nn.GELU(),

        )

        # 位置编码
        self.register_buffer('pos_encoding', self._create_pos_encoding(64, 64))

    def _create_pos_encoding(self, H, W):
        """创建位置编码"""
        pos_encoding = torch.zeros(1, 1, H, W)
        for i in range(H):
            for j in range(W):
                pos_encoding[0, 0, i, j] = (i / H + j / W) / 2
        return pos_encoding


    def forward(self, x, return_indices=True):
        B, C, H, W = x.shape
        L = H * W

        # 调整位置编码大小
        if H != self.pos_encoding.shape[2] or W != self.pos_encoding.shape[3]:
            pos_encoding = F.interpolate(self.pos_encoding, size=(H, W), mode='bilinear')
        else:
            pos_encoding = self.pos_encoding

        # 添加位置信息
        x_with_pos = x + pos_encoding.expand(B, -1, -1, -1) * 0.1

        # 预测每个位置的扫描分数
        attention_scores = self.attention_predictor(x_with_pos)  # (B, num_directions, H, W)

        scan_indices_list = []
        x_scanned_list = []

        for d in range(self.num_directions):
            direction_scores = attention_scores[:, d].view(B, -1)

            # 获取排序索引
            _, scan_indices = torch.sort(direction_scores, dim=1, descending=True)

            # 更安全的重排方式
            x_flat = x.view(B, C, L)
            x_scanned = torch.zeros_like(x_flat)

            # 使用循环避免复杂的索引操作
            for b in range(B):
                x_scanned[b] = x_flat[b, :, scan_indices[b]]

            scan_indices_list.append(scan_indices)
            x_scanned_list.append(x_scanned)

        if return_indices:
            return x_scanned_list, scan_indices_list
        return x_scanned_list


import torch
import torch.nn as nn


class GroupNorm(nn.GroupNorm):
    """
    使用1个组进行的Group Normalization。
    输入：形状为[B, C, H, W]的张量

    参数：
        num_channels: 输入通道数
        **kwargs: 传递给父类的其他参数
    """

    def __init__(self, num_channels, **kwargs):
        # 初始化时指定1个组和传入的通道数
        super().__init__(1, num_channels, **kwargs)


class Conv2d_BN(nn.Sequential):
    """
    卷积层 + BatchNorm2d 的组合层

    参数：
        a: 输入通道数
        b: 输出通道数
        ks: 卷积核大小，默认为1
        stride: 步长，默认为1
        pad: 填充，默认为0
        dilation: 膨胀率，默认为1
        groups: 分组卷积的组数，默认为1
        bn_weight_init: BatchNorm权重初始化值，默认为1
    """

    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()

        # 添加卷积层（无偏置）
        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False
        ))

        # 添加BatchNorm层
        self.add_module('bn', nn.BatchNorm2d(b))

        # 初始化BatchNorm参数
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        """
        融合卷积层和BatchNorm层

        返回：
            nn.Conv2d: 融合后的卷积层
        """
        c, bn = self._modules.values()

        # 计算融合后的权重
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]

        # 计算融合后的偏置
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5

        # 创建新的卷积层
        m = nn.Conv2d(
            in_channels=w.size(1) * c.groups,
            out_channels=w.size(0),
            kernel_size=w.shape[2:],
            stride=c.stride,
            padding=c.padding,
            dilation=c.dilation,
            groups=c.groups,
            device=c.weight.device
        )

        # 设置权重和偏置
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)

        return m


import torch
import torch.nn as nn




class SS2D(nn.Module, SS2Dv2, SS2Dm0):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU(inplace=False),
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            initialize="v0",
            # ======================
            forward_type="v5",
            channel_first=False,
            # ======================
            k_group=4,
            # 注意力扫描新增参数 =====
            use_attention_scan=False,  # 新增：是否使用注意力扫描
            attention_type="simam",  # "simam" 或 "learnable"
            # ======================
            **kwargs,
    ):
        super().__init__()

        # 保存注意力扫描参数
        self.use_attention_scan = use_attention_scan
        self.attention_type = attention_type

        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first, k_group=k_group,
        )

        if forward_type in ["v0", "v0seq"]:
            self.__initv0__(seq=("seq" in forward_type), **kwargs)
        elif forward_type.startswith("xv"):
            self.__initxv__(**kwargs)
        elif forward_type.startswith("m"):
            self.__initm0__(**kwargs)
        else:
            self.__initv2__(**kwargs)

        # 如果需要注意力扫描，初始化相关模块
        if self.use_attention_scan:
            self._init_attention_modules(**kwargs)

        # 如果forward_type是v052d，使用注意力扫描版本
        # if forward_type == "v052d":
        #     self.forward_core = self.forward_core_with_attention

    def _init_attention_modules(self, **kwargs):
        """初始化注意力相关模块"""
        d_model = kwargs.get('d_model', 96)
        k_group = kwargs.get('k_group', 4)

        if self.attention_type == "simam":
            #self.simam = SIMAM(d_model)
            self.simam = nn.Sequential(
            SIMAM(d_model),

        )

        elif self.attention_type == "learnable":
            self.attention_scan = EfficientAttentionScan(d_model, num_directions=k_group)

        # 保存原始的forward_core函数
        if hasattr(self, 'forward_core'):
            self._original_forward_core = self.forward_core

    def compute_attention_scan_indices(self, x):
        """计算注意力扫描的索引"""
        if self.attention_type == "simam":
            # 使用SIMAM计算注意力
            with torch.no_grad():
                attn_out = self.simam(x)  # (B, C, H, W)

                # 获取通道分组信息
                B, C, H, W = attn_out.shape
                channels_per_group = C // self.k_group  # 每个组的通道数

                scan_indices_list = []

                for k in range(self.k_group):
                    # 获取当前组的通道范围
                    start_channel = k * channels_per_group
                    end_channel = (k + 1) * channels_per_group

                    # 提取当前组的注意力图并计算平均值
                    group_attention = attn_out[:, start_channel:end_channel, :, :]  # (B, C_per_group, H, W)
                    attention_map = group_attention.mean(dim=1, keepdim=True)  # (B, 1, H, W)

                        # 第一个方向：按注意力降序
                    scores = attention_map.view(B, -1)

                    _, indices = torch.sort(scores, dim=1, descending=True)
                    scan_indices_list.append(indices)

                return scan_indices_list

        elif self.attention_type == "learnable":
            # 使用可学习的注意力扫描
            _, scan_indices_list = self.attention_scan(x)
            return scan_indices_list

        return None

    def apply_attention_scan(self, x, scan_indices_list):
        """应用注意力扫描 - 修复版本"""
        B, C, H, W = x.shape
        L = H * W
        K = len(scan_indices_list)

        # 准备结果张量
        x_scanned_all = torch.zeros(B, K, C, L, device=x.device, dtype=x.dtype)

        # 对每个扫描方向应用重排
        for k in range(K):
            scan_indices = scan_indices_list[k]  # shape: (B, L)
            x_flat = x.view(B, C, L)  # shape: (B, C, L)

            # 使用gather进行高效的重排
            # 扩展索引以匹配x_flat的形状
            expanded_indices = scan_indices.unsqueeze(1).expand(-1, C, -1)  # (B, C, L)

            # 使用gather进行重排
            x_scanned = torch.gather(x_flat, dim=2, index=expanded_indices)
            x_scanned_all[:, k] = x_scanned

        return x_scanned_all

    def restore_from_attention_scan(self, y_scanned_all, scan_indices_list):
        """从注意力扫描结果恢复原始顺序 - 修复版本"""
        B, K, C, L = y_scanned_all.shape
        H = W = int(L ** 0.5)

        # 合并所有方向的结果
        y_restored_all = torch.zeros(B, K, C, L, device=y_scanned_all.device)

        for k in range(K):
            scan_indices = scan_indices_list[k]  # (B, L)

            # 创建反向索引
            inverse_indices = torch.argsort(scan_indices, dim=1)  # (B, L)

            # 扩展反向索引以匹配y_scanned_all的形状
            expanded_inverse = inverse_indices.unsqueeze(1).expand(-1, C, -1)  # (B, C, L)

            # 使用scatter进行反向重排
            y_restored = torch.zeros(B, C, L, device=y_scanned_all.device)
            y_restored.scatter_(dim=2, index=expanded_inverse, src=y_scanned_all[:, k])
            y_restored_all[:, k] = y_restored

        # 合并所有方向（简单求和）
        y_restored = y_restored_all.sum(dim=1).view(B, C, H, W)

        return y_restored

    def forward_core_with_attention(self, x, **kwargs):
        """支持注意力扫描的forward_core"""
        if not self.use_attention_scan:
            return self._original_forward_core(x, **kwargs)

        B, D, H, W = x.shape
        L = H * W

        # 1. 计算注意力扫描索引
        scan_indices_list = self.compute_attention_scan_indices(x)

        if scan_indices_list is None:
            # 回退到原始实现
            return self._original_forward_core(x, **kwargs)

        # 2. 应用扫描
        x_scanned_all = self.apply_attention_scan(x, scan_indices_list)  # (B, K, D, L)

        # 3. 对每个扫描方向进行处理
        # 这里需要调用原始的状态空间计算
        # 由于原SS2D处理的是4D输入，我们需要调整

        # 临时解决方案：将扫描后的序列视为batch维度
        B_total = B * self.k_group
        x_processed_all = x_scanned_all.permute(0, 1, 3, 2).contiguous()  # (B, K, L, D)
        x_processed_all = x_processed_all.view(B_total, L, D)

        # 这里需要调用1D版本的状态空间计算
        # 由于原SS2D是2D的，我们需要简化处理
        y_processed_all = self._simplified_1d_ssm(x_processed_all)

        # 恢复形状
        y_processed_all = y_processed_all.view(B, self.k_group, L, D).permute(0, 1, 3, 2)  # (B, K, D, L)

        # 4. 恢复原始顺序
        y_restored = self.restore_from_attention_scan(y_processed_all, scan_indices_list)

        return y_restored

    def _simplified_1d_ssm(self, x):
        """简化的1D状态空间计算（用于演示）"""
        # 在实际应用中，这里应该调用真正的1D SSM
        # 这里使用简单的投影作为演示
        B, L, D = x.shape

        # 简化的状态空间计算
        # 实际应该使用：y = selective_scan(x, ...)
        # 这里暂时使用恒等映射
        y = x

        return y

    def forwardv2(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))  # (b, h, w, d)
        z = self.act(z.clone())
        x = self.conv2d(x) if hasattr(self, 'conv2d') else x
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out







# =====================================================
class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
            channel_first=False,
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=1,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU(inplace=False),
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v05_noz",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            gmlp=False,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )

        self.drop_path = DropPath(drop_path)

        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = _MLP(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                            drop=mlp_drop_rate, channels_first=channel_first)

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))  # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class VSSM(nn.Module):
    def __init__(
            self,
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 2, 9, 2],
            dims=[96, 192, 384, 768],
            # =========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # =========================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            gmlp=False,
            # =========================
            drop_path_rate=0.1,
            patch_norm=True,
            norm_layer="LN",  # "BN", "LN2D"
            downsample_version: str = "v2",  # "v1", "v2", "v3"
            patchembed_version: str = "v1",  # "v1", "v2"
            use_checkpoint=False,
            # =========================
            posembed=False,
            imgsize=224,
            **kwargs,
    ):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU(inplace=False),
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.pos_embed = self._pos_embed(dims[0], patch_size, imgsize) if posembed else None

        _make_patch_embed = dict(
            v1=self._make_patch_embed,
            v2=self._make_patch_embed_v2,
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer,
                                             channel_first=self.channel_first)

        _make_downsample = dict(
            v1=PatchMerging2D,
            v2=self._make_downsample,
            v3=self._make_downsample_v3,
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = _make_downsample(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                norm_layer=norm_layer,
                channel_first=self.channel_first,
            ) if (i_layer < self.num_layers - 1) else nn.Identity()

            self.layers.append(self._make_layer(
                dim=self.dims[i_layer],
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                downsample=downsample,
                channel_first=self.channel_first,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features),  # B,H,W,C
            permute=(Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        patch_height, patch_width = (img_size // patch_size, img_size // patch_size)
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, patch_height, patch_width))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    # used in building optimizer
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    @staticmethod
    def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                          channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm,
                             channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            channel_first=False,
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer=nn.SiLU(inplace=False),
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v2",
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
            **kwargs,
    ):
        # if channel first, then Norm and Output are both channel_first
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=nn.SiLU(inplace=False),
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
            ))

        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks, ),
            downsample=downsample,
        ))

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        out_features = []
        if self.pos_embed is not None:
            pos_embed = self.pos_embed.permute(0, 2, 3, 1) if not self.channel_first else self.pos_embed
            x = x + pos_embed
        out_features.append(x)
        for layer in self.layers:
            x = layer(x)
            if len(out_features) < 2:
                out_features.append(x)
        x = self.classifier(x)
        return x

    def flops(self, shape=(3, 224, 224), verbose=True):
        # shape = self.__input_shape__[1:]
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanCuda": partial(selective_scan_flop_jit, backend="prefixsum", verbose=verbose),
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):

        def check_name(src, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    return True
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        return True
            return False

        def change_name(src, dst, state_dict: dict = state_dict, strict=False):
            if strict:
                if prefix + src in list(state_dict.keys()):
                    state_dict[prefix + dst] = state_dict[prefix + src]
                    state_dict.pop(prefix + src)
            else:
                key = prefix + src
                for k in list(state_dict.keys()):
                    if k.startswith(key):
                        new_k = prefix + dst + k[len(key):]
                        state_dict[new_k] = state_dict[k]
                        state_dict.pop(k)

        if check_name("pos_embed", strict=True):
            srcEmb: torch.Tensor = state_dict[prefix + "pos_embed"]
            state_dict[prefix + "pos_embed"] = F.interpolate(srcEmb.float(), size=self.pos_embed.shape[2:4],
                                                             align_corners=False, mode="bicubic").to(srcEmb.device)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
                change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)


# compatible with openmmlab
class Backbone_VSSM(VSSM):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, norm_layer="ln", **kwargs):
        kwargs.update(norm_layer=norm_layer)
        super().__init__(**kwargs)
        self.channel_first = (norm_layer.lower() in ["bn", "ln2d"])
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        norm_layer: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.classifier
        self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")

    def forward(self, x):
        def layer_forward(l, x):
            x = l.blocks(x)
            y = l.downsample(x)
            return x, y

        x = self.patch_embed(x)
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x)  # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)
                outs.append(out.contiguous())

        if len(self.out_indices) == 0:
            return x

        return outs


# =====================================================
def vanilla_vmamba_tiny():
    return VSSM(
        depths=[2, 2, 9, 2], dims=96, drop_path_rate=0.2,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v0",
        mlp_ratio=0.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v1", patchembed_version="v1",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


def vanilla_vmamba_small():
    return VSSM(
        depths=[2, 2, 27, 2], dims=96, drop_path_rate=0.3,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v0",
        mlp_ratio=0.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v1", patchembed_version="v1",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


def vanilla_vmamba_base():
    return VSSM(
        depths=[2, 2, 27, 2], dims=128, drop_path_rate=0.6,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=16, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=True, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v0",
        mlp_ratio=0.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v1", patchembed_version="v1",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


# =====================================================
def vmamba_tiny_s2l5(channel_first=True):
    return VSSM(
        depths=[2, 2, 5, 2], dims=96, drop_path_rate=0.2,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v05_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


def vmamba_small_s2l15(channel_first=True):
    return VSSM(
        depths=[2, 2, 15, 2], dims=96, drop_path_rate=0.3,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v05_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


def vmamba_base_s2l15(channel_first=True):
    return VSSM(
        depths=[2, 2, 15, 2], dims=128, drop_path_rate=0.6,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v05_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


# =====================================================
def vmamba_tiny_s1l8(channel_first=True):
    return VSSM(
        depths=[2, 2, 8, 2], dims=96, drop_path_rate=0.2,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v05_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


def vmamba_small_s1l20(channel_first=True):
    return VSSM(
        depths=[2, 2, 20, 2], dims=96, drop_path_rate=0.3,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v05_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


def vmamba_base_s1l20(channel_first=True):
    return VSSM(
        depths=[2, 2, 20, 2], dims=128, drop_path_rate=0.5,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=1, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="silu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v0", forward_type="v05_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer=("ln2d" if channel_first else "ln"),
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


# mamba2 support =====================================================
def vmamba_tiny_m2():
    return VSSM(
        depths=[2, 2, 4, 2], dims=96, drop_path_rate=0.2,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v2", forward_type="m0_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


def vmamba_small_m2():
    return VSSM(
        depths=[2, 2, 12, 2], dims=96, drop_path_rate=0.3,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v2", forward_type="m0_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


def vmamba_base_m2():
    return VSSM(
        depths=[2, 2, 12, 2], dims=128, drop_path_rate=0.3,
        patch_size=4, in_chans=3, num_classes=1000,
        ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
        ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
        ssm_init="v2", forward_type="m0_noz",
        mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
        patch_norm=True, norm_layer="ln",
        downsample_version="v3", patchembed_version="v2",
        use_checkpoint=False, posembed=False, imgsize=224,
    )


if __name__ == "__main__":
    model = vmamba_tiny_s1l8()

    # model = VSSM(
    #     depths=[2, 2, 4, 2], dims=96, drop_path_rate=0.2,
    #     patch_size=4, in_chans=3, num_classes=1000,
    #     ssm_d_state=64, ssm_ratio=1.0, ssm_dt_rank="auto", ssm_act_layer="gelu",
    #     ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0,
    #     ssm_init="v2", forward_type="m0_noz",
    #     mlp_ratio=4.0, mlp_act_layer="gelu", mlp_drop_rate=0.0, gmlp=False,
    #     patch_norm=True, norm_layer="ln",
    #     downsample_version="v3", patchembed_version="v2",
    #     use_checkpoint=False, posembed=False, imgsize=224,
    # )
    # print(parameter_count(model)[""])
    # print(model.flops()) # wrong
    # model.cuda().train()
    model_weights_path = 'vssm1_tiny_0230s_ckpt_epoch_264.pth'
    checkpoint = torch.load(model_weights_path, map_location='cpu')
    # if 'model' in checkpoint:
    #     msg = model.load_state_dict(checkpoint['model'], strict=False)
    #     print(msg)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda().eval()
    x = torch.randn(1, 3, 256, 256).cuda()
    y, features = model(x)
    print('finish')


    def bench(model):
        import time
        inp = torch.randn((128, 3, 224, 224)).cuda()
        for _ in range(30):
            model(inp)
        torch.cuda.synchronize()
        tim = time.time()
        for _ in range(30):
            model(inp)
        torch.cuda.synchronize()
        tim1 = time.time() - tim

        for _ in range(30):
            model(inp).sum().backward()
        torch.cuda.synchronize()
        tim = time.time()
        for _ in range(30):
            model(inp).sum().backward()
        torch.cuda.synchronize()
        tim2 = time.time() - tim

        return tim1 / 30, tim2 / 30

    # print(bench(model_ref))
    # print(bench(model))
    #
    # breakpoint()


