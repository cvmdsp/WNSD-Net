# @Time    : 2023/3/17 15:56
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : ABCNet.py
# @Software: PyCharm
import torch
import torch.nn as nn
from einops import rearrange
from .gcn.layers.GConv2 import GConv2

from pytorch_wavelets import DWTForward, DWTInverse

import torch.nn.functional as F

from .crossformer_backbone_det import CrossFormerBlock,CrossFormerBlock1,CrossFormerBlock2,CrossFormerBlock3





class CWST2(nn.Module):
    def __init__(self, in_dim, out_dim, size):
        super(CWST2, self).__init__()
        self.convs = GConv2(int((in_dim/1) / 4), int((in_dim/1) / 4), 3, M=4, padding=1)



    def forward(self, x):

        convsy = self.convs(x)

        out = convsy
        return out


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


class Gabor(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(Gabor, self).__init__()
        self.dim = dim
        self.dim_sp = dim
        # PW first or DW first?
        self.conv_init = nn.Sequential(  # PW->DW->
            nn.Conv2d(dim, dim*2, 1),
        )

        self.conv1_1 = nn.Sequential(
            GConv2(int((self.dim_sp/1) / 4), int((self.dim_sp/1) / 4), 3,nScale=3, M=4, padding=1),
        )
        self.conv1_2 = nn.Sequential(
            GConv2(int((self.dim_sp/1) / 4), int((self.dim_sp/1) / 4), 5,nScale=4, M=4, padding=2),
        )
        self.conv1_3 = nn.Sequential(
            GConv2(int((self.dim_sp/1) / 4), int((self.dim_sp/1) / 4), 7,nScale=5,M=4, padding=3),
        )




    def forward(self, x):

        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x1)
        x3 = self.conv1_3(x2)+x


        return x3



class CWST3(nn.Module):
    def __init__(self, in_dim, interval, size):
        super(CWST3, self).__init__()
        self.WTF = DWTForward(J=1, wave="haar")
        self.WTI = DWTInverse(mode='zero', wave="haar")
        self.wdim = [in_dim, in_dim]


        self.trans1 = CrossFormerBlock1(dim=int(in_dim//4), num_heads=4, group_size=size, interval=interval,lsda_flag=1)
        self.convs =  GConv2(int((in_dim/1) / 4), int((in_dim/1) / 4), 3, M=4, padding=1)
        self.fusion = ChannelSelect(in_dim)

    def forward(self, x):
        # padding = 0 #haar
        # padding=3 #db3
        # padding = 2  # db4
        padding = 0  # db2

        x_=x
        convsy = self.convs(x)
        x = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)

        xl1, xh1 = self.WTF(x)

        b, c, n, h, w = xh1[0].shape
        channels_per_group = xh1[0].shape[2] // 3
        s1, s2, s3 = torch.split(xh1[0], channels_per_group, 2)
        s1 = s1.view(b, -1, h, w)
        s2 = s2.view(b, -1, h, w)
        s3 = s3.view(b, -1, h, w)

        top = torch.cat((xl1, s1), dim=3)  # dim=3 是宽度的拼接
        bottom = torch.cat((s2, s3), dim=3)  # dim=3 是宽度的拼接

        # 最后在批次维度上拼接 top 和 bottom
        feature_hl = torch.cat((top, bottom), dim=2)  # dim=2 是高度的拼接
        feature_hl = feature_hl.view(b, -1, c)

        #trans_fea2 = self.trans1(feature_hl,2*h,2*w).view(b,int(c//1),2*h,2*w)

        channels_per_group1 = feature_hl.shape[2] // 4
        feature_hl_1, feature_hl_2, feature_hl_3, feature_hl_4= torch.split(feature_hl, channels_per_group1, 2)

        trans_fea11 = self.trans1(feature_hl_1,2*h,2*w).view(b,int(c//4),2*h,2*w)
        trans_fea12 = self.trans1(feature_hl_2,2*h,2*w).view(b,int(c//4),2*h,2*w)
        trans_fea13 = self.trans1(feature_hl_3,2*h,2*w).view(b,int(c//4),2*h,2*w)
        trans_fea14 = self.trans1(feature_hl_4,2*h,2*w).view(b,int(c//4),2*h,2*w)


        trans_fea2 = torch.cat((trans_fea11, trans_fea12,trans_fea13, trans_fea14), dim=1)


        yl1 = trans_fea2[:, :, :h, :w] # 左上角+
        x2_new = trans_fea2[:, :, :h, w:] # 右上角
        x3_new = trans_fea2[:, :, h:, :w] # 左下角
        x4_new = trans_fea2[:, :, h:, w:] # 右下角


        yh1 = torch.stack(
            [x2_new, x3_new, x4_new], dim=2
        )
        # # #
        yh1_ = [None]
        yh1_[0] = yh1.view(b, -1, 3, h, w)


        transy = self.WTI((yl1, yh1_))



        # transy = transy[:, :, padding:-1*padding, padding:-1*padding]  # 去掉上下左右各 4

        out = self.fusion(transy,convsy)+x_
        return out

class CWST4(nn.Module):
    def __init__(self, in_dim, interval, size):
        super(CWST4, self).__init__()
        self.WTF = DWTForward(J=1, wave="haar")
        self.WTI = DWTInverse(mode='zero', wave="haar")
        self.wdim = [in_dim, in_dim]


        self.trans1 = CrossFormerBlock2(dim=int(in_dim//4), num_heads=2, group_size=size, interval=interval,lsda_flag=1)
        self.convs =  GConv2(int((in_dim/1) / 4), int((in_dim/1) / 4), 3, M=4, padding=1)
        self.fusion = ChannelSelect(in_dim)

    def forward(self, x):
        # padding = 0 #haar
        # padding=3 #db3
        # padding = 2  # db4
        padding = 0  # db2

        x_=x
        convsy = self.convs(x)
        x = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)

        xl1, xh1 = self.WTF(x)

        b, c, n, h, w = xh1[0].shape
        channels_per_group = xh1[0].shape[2] // 3
        s1, s2, s3 = torch.split(xh1[0], channels_per_group, 2)
        s1 = s1.view(b, -1, h, w)
        s2 = s2.view(b, -1, h, w)
        s3 = s3.view(b, -1, h, w)

        top = torch.cat((xl1, s1), dim=3)  # dim=3 是宽度的拼接
        bottom = torch.cat((s2, s3), dim=3)  # dim=3 是宽度的拼接

        # 最后在批次维度上拼接 top 和 bottom
        feature_hl = torch.cat((top, bottom), dim=2)  # dim=2 是高度的拼接
        feature_hl = feature_hl.view(b, -1, c)

        #trans_fea2 = self.trans1(feature_hl,2*h,2*w).view(b,int(c//1),2*h,2*w)

        channels_per_group1 = feature_hl.shape[2] // 4
        feature_hl_1, feature_hl_2, feature_hl_3, feature_hl_4= torch.split(feature_hl, channels_per_group1, 2)

        trans_fea11 = self.trans1(feature_hl_1,2*h,2*w).view(b,int(c//4),2*h,2*w)
        trans_fea12 = self.trans1(feature_hl_2,2*h,2*w).view(b,int(c//4),2*h,2*w)
        trans_fea13 = self.trans1(feature_hl_3,2*h,2*w).view(b,int(c//4),2*h,2*w)
        trans_fea14 = self.trans1(feature_hl_4,2*h,2*w).view(b,int(c//4),2*h,2*w)


        trans_fea2 = torch.cat((trans_fea11, trans_fea12,trans_fea13, trans_fea14), dim=1)


        yl1 = trans_fea2[:, :, :h, :w] # 左上角+
        x2_new = trans_fea2[:, :, :h, w:] # 右上角
        x3_new = trans_fea2[:, :, h:, :w] # 左下角
        x4_new = trans_fea2[:, :, h:, w:] # 右下角


        yh1 = torch.stack(
            [x2_new, x3_new, x4_new], dim=2
        )
        # # #
        yh1_ = [None]
        yh1_[0] = yh1.view(b, -1, 3, h, w)


        transy = self.WTI((yl1, yh1_))



        # transy = transy[:, :, padding:-1*padding, padding:-1*padding]  # 去掉上下左右各 4

        out = self.fusion(transy,convsy)+x_
        return out




class CWST5(nn.Module):
    def __init__(self, in_dim, interval, size):
        super(CWST5, self).__init__()
        self.WTF = DWTForward(J=1, wave="db2")
        self.WTI = DWTInverse(mode='zero', wave="db2")
        self.wdim = [in_dim, in_dim]


        self.trans1 = CrossFormerBlock3(dim=int(in_dim//4), num_heads=2, group_size=size, interval=interval,lsda_flag=1)
        self.convs =  GConv2(int((in_dim/1) / 4), int((in_dim/1) / 4), 3, M=4, padding=1)
        self.fusion = ChannelSelect(in_dim)

    def forward(self, x):
        # padding = 0 #haar
        # padding=3 #db3
        # padding = 2  # db4
        padding = 3  # db2

        x_=x
        convsy = self.convs(x)
        x = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)

        xl1, xh1 = self.WTF(x)

        b, c, n, h, w = xh1[0].shape
        channels_per_group = xh1[0].shape[2] // 3
        s1, s2, s3 = torch.split(xh1[0], channels_per_group, 2)
        s1 = s1.view(b, -1, h, w)
        s2 = s2.view(b, -1, h, w)
        s3 = s3.view(b, -1, h, w)

        top = torch.cat((xl1, s1), dim=3)  # dim=3 是宽度的拼接
        bottom = torch.cat((s2, s3), dim=3)  # dim=3 是宽度的拼接

        # 最后在批次维度上拼接 top 和 bottom
        feature_hl = torch.cat((top, bottom), dim=2)  # dim=2 是高度的拼接
        feature_hl = feature_hl.view(b, -1, c)

        #trans_fea2 = self.trans1(feature_hl,2*h,2*w).view(b,int(c//1),2*h,2*w)

        channels_per_group1 = feature_hl.shape[2] // 4
        feature_hl_1, feature_hl_2, feature_hl_3, feature_hl_4= torch.split(feature_hl, channels_per_group1, 2)

        trans_fea11 = self.trans1(feature_hl_1,2*h,2*w).view(b,int(c//4),2*h,2*w)
        trans_fea12 = self.trans1(feature_hl_2,2*h,2*w).view(b,int(c//4),2*h,2*w)
        trans_fea13 = self.trans1(feature_hl_3,2*h,2*w).view(b,int(c//4),2*h,2*w)
        trans_fea14 = self.trans1(feature_hl_4,2*h,2*w).view(b,int(c//4),2*h,2*w)


        trans_fea2 = torch.cat((trans_fea11, trans_fea12,trans_fea13, trans_fea14), dim=1)


        yl1 = trans_fea2[:, :, :h, :w] # 左上角+
        x2_new = trans_fea2[:, :, :h, w:] # 右上角
        x3_new = trans_fea2[:, :, h:, :w] # 左下角
        x4_new = trans_fea2[:, :, h:, w:] # 右下角


        yh1 = torch.stack(
            [x2_new, x3_new, x4_new], dim=2
        )
        # # #
        yh1_ = [None]
        yh1_[0] = yh1.view(b, -1, 3, h, w)


        transy = self.WTI((yl1, yh1_))



        transy = transy[:, :, padding:-1*padding, padding:-1*padding]  # 去掉上下左右各 4

        out = self.fusion(transy,convsy)+x_
        return out


# class CWST2(nn.Module):
#     def __init__(self, in_dim, out_dim, size):
#         super(CWST2, self).__init__()
#         self.WTF = DWTForward(J=1, wave="haar")
#         self.WTI = DWTInverse(mode='zero', wave="haar")
#         self.wdim = [in_dim, in_dim]
#
#         len = size
#         #interval = len+5 except haar
#         self.trans1 = CrossFormerBlock1(dim=int(in_dim/4), input_resolution=[len*10, len*10], num_heads=1, group_size=size, interval=len,
#                                        lsda_flag=1)
#         # self.trans1 = CrossFormerBlock1(dim=int(in_dim/4), input_resolution=[len, len], num_heads=1, group_size=2, interval=len+5,
#         #                                lsda_flag=1)
#         self.convs = GConv2(int((in_dim/1) / 4), int((in_dim/1) / 4), 5, M=4, padding=2)
#
#         self.fusion = ChannelSelect(in_dim)
#
#     def forward(self, x):
#
#         # padding=3 #db3
#         # padding = 2  # db4
#         # padding = 4  # db2
#         padding = 0
#         x_=x
#         x = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
#
#         xl1, xh1 = self.WTF(x)
#
#         b, c, n, h, w = xh1[0].shape
#         channels_per_group = xh1[0].shape[2] // 3
#         s1, s2, s3 = torch.split(xh1[0], channels_per_group, 2)
#         s1 = s1.view(b, -1, h, w)
#         s2 = s2.view(b, -1, h, w)
#         s3 = s3.view(b, -1, h, w)
#
#         top = torch.cat((xl1, s1), dim=3)  # dim=3 是宽度的拼接
#         bottom = torch.cat((s2, s3), dim=3)  # dim=3 是宽度的拼接
#
#         # 最后在批次维度上拼接 top 和 bottom
#         feature_hl = torch.cat((top, bottom), dim=2)  # dim=2 是高度的拼接
#         feature_hl = feature_hl.view(b, -1, c)
#
#         channels_per_group1 = feature_hl.shape[2] // 4
#         feature_hl_1, feature_hl_2, feature_hl_3, feature_hl_4= torch.split(feature_hl, channels_per_group1, 2)
#
#         trans_fea11 = self.trans1(feature_hl_1,2*h,2*w).view(b,int(c//4),2*h,2*w)
#         trans_fea12 = self.trans1(feature_hl_2,2*h,2*w).view(b,int(c//4),2*h,2*w)
#         trans_fea13 = self.trans1(feature_hl_3,2*h,2*w).view(b,int(c//4),2*h,2*w)
#         trans_fea14 = self.trans1(feature_hl_4,2*h,2*w).view(b,int(c//4),2*h,2*w)
#
#
#         trans_fea2 = torch.cat((trans_fea11, trans_fea12, trans_fea13,trans_fea14), dim=1)
#
#         yl1 = trans_fea2[:, :, :h, :w] + xl1  # 左上角+
#         x2_new = trans_fea2[:, :, :h, w:] + s1  # 右上角
#         x3_new = trans_fea2[:, :, h:, :w] + s2  # 左下角
#         x4_new = trans_fea2[:, :, h:, w:] + s3  # 右下角
#
#
#         yh1 = torch.stack(
#             [x2_new, x3_new, x4_new], dim=2
#         )
#         # # #
#         yh1_ = [None]
#         yh1_[0] = yh1.view(b, -1, 3, h, w)
#
#         # transy = self.WTI((yl1, xh1))
#         transy = self.WTI((yl1, yh1_))
#         convsy = self.convs(x_)
#         # transy = transy[:, :, padding:-1*padding, padding:-1*padding]  # 去掉上下左右各 4
#
#         out = self.fusion(transy, convsy)+x_
#         return out
#
#








class Attention(nn.Module):
    def __init__(self, in_dim, in_feature, out_feature):
        super(Attention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.query_line = nn.Linear(in_features=in_feature, out_features=out_feature)
        self.key_line = nn.Linear(in_features=in_feature, out_features=out_feature)
        self.s_conv = nn.Conv2d(in_channels=1, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = rearrange(self.query_line(rearrange(self.query_conv(x), 'b 1 h w -> b (h w)')), 'b h -> b h 1')
        k = rearrange(self.key_line(rearrange(self.key_conv(x), 'b 1 h w -> b (h w)')), 'b h -> b 1 h')
        att = rearrange(torch.matmul(q, k), 'b h w -> b 1 h w')
        att = self.softmax(self.s_conv(att))
        return att


class DEPTHWISECONV(nn.Module):
    def __init__(self,in_ch,out_ch,kernel):
        super(DEPTHWISECONV, self).__init__()


        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=kernel,
                                    stride=1,
                                    padding="same",
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class Conv2(nn.Module):
    def __init__(self, in_dim):
        super(Conv2, self).__init__()
        # self.convs = nn.ModuleList([conv_relu_bn(in_dim, in_dim, 1) for _ in range(3)])
        self.convs = DEPTHWISECONV(in_dim,in_dim,3,padding=1)

    def forward(self, x):
        # for conv in self.convs:
        #     x = conv(x)
        x = self.convs(x)
        return x

class Conv(nn.Module):
    def __init__(self, in_dim):
        super(Conv, self).__init__()
        #self.convs = nn.ModuleList([conv_relu_bn(in_dim, in_dim, 1) for _ in range(3)])
        #self.convs =  DEPTHWISECONV(in_dim,in_dim,3,padding=1)
        #self.convs = GaborFPU(int(in_dim),int(in_dim))
        #self.convs = nn.Conv2d(in_dim,in_dim,1,1)
        self.convs = GConv2(int(in_dim/4), int(in_dim/4), 3,M=4, padding=1)

    def forward(self, x):
        # for conv in self.convs:
        #     x = conv(x)
        x = self.convs(x)
        # x = self.convs(x)
        return x








class ConvTransformer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ConvTransformer, self).__init__()

        # self.feedforward = FeedForward(in_dim, out_dim)

        # self.dim_cwt = int(in_dim // 1)
        # self.dim_untouched = in_dim - self.dim_cwt
        self.attention = DEPTHWISECONV(in_dim,out_dim,3)
    def forward(self, x):
        #xa, xna = torch.split(x, [self.dim_cwt, self.dim_untouched], dim=1)

        x = self.attention(x)
        # out = self.feedforward(x)
        #out = torch.cat((xa, xna), 1)

        return x

# def norm_tensor(x):
#     min_value = x.min()
#     max_value = x.max()
#     out = (x - min_value) / (max_value - min_value)
#     return out







def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p








#
# class ChannelSelect2(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelSelect2, self).__init__()
#
#         # self.advavg = nn.AdaptiveAvgPool2d(1)
#         # self.conv = DEPTHWISECONV(in_planes*2,in_planes*2,1)
#
#         self.conv1 = nn.Conv2d(in_planes, in_planes, 1)
#         self.conv2 = nn.Conv2d(in_planes, in_planes, 1)
#         self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
#     def forward(self, x, y):
#
#         attn1 = self.conv1(x)
#         attn2 = self.conv2(y)
#         attn = torch.cat([attn1, attn2], dim=1)
#         avg_attn = torch.mean(attn, dim=1, keepdim=True)
#         max_attn, _ = torch.max(attn, dim=1, keepdim=True)
#         agg = torch.cat([avg_attn, max_attn], dim=1)
#         sig = self.conv_squeeze(agg).sigmoid()
#         output = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)
#         # attn = self.conv(attn)
#         # return x * attn
#
#         return output



class ChannelSelect(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelSelect, self).__init__()

        self.advavg = nn.AdaptiveAvgPool2d(1)
        self.conv = DEPTHWISECONV(in_planes*2,in_planes*2,1)
        # self.conv2 = DEPTHWISECONV(in_planes, in_planes, 3)
    def forward(self, x, y):
        # b,c,h,w = x.shape
        # xfc = x.view(b,c,-1)
        # yfc = y.view(b,c,-1)
        xy = torch.cat([x, y], dim=1)
        ca = F.softmax(self.conv(self.advavg(xy)), dim=1).sigmoid()
        cax,cay = torch.split(ca, ca.size(1) // 2, dim=1)

        mask1 = cax > cay  # 如果 score1 大于 score2，选择 x1
        mask2 = cax < cay  # 如果 score1 小于 score2，选择 x2
        mask_equal = cax == cay  # 如果 score1 等于 score2，选择 x1 + x2
        # 通过掩膜选择特征图
        output = torch.zeros_like(x)  # 初始化输出
        output += mask1 * x  # score1 > score2 时选择 x1
        output += mask2 * y  # score1 < score2 时选择 x2
        output += mask_equal * ((x + y)/2)  # score1 == score2 时选择 x1 + x2
        # output = self.conv2(output)
        return output



class ChannelSelect2(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelSelect2, self).__init__()

        self.advavg = nn.AdaptiveAvgPool2d(1)
        self.conv = DEPTHWISECONV(in_planes*2,in_planes*2,1)
    def forward(self, x, y):
        # b,c,h,w = x.shape
        # xfc = x.view(b,c,-1)
        # yfc = y.view(b,c,-1)
        xy = torch.cat([x, y], dim=1)
        ca = F.softmax(self.conv(self.advavg(xy)), dim=1).sigmoid()
        cax,cay = torch.split(ca, ca.size(1) // 2, dim=1)

        output = x*cax+y*cay

        return output









