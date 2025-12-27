import os
import re
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
from collections import OrderedDict
from util.common import ScaleInOutput
from dropblock import LinearScheduler, DropBlock2D
from pytorch_wavelets import DWTForward, DWTInverse
from models.cswin_DWT import WTConv, CSWin_64_12211_tiny_224, CSWin_64_24322_small_224, CSWin_96_24322_base_384, CSWin_96_24322_base_224



class Conv3Relu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Conv3Relu, self).__init__()
        self.extract = nn.Sequential(nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1),
                                               stride=(stride, stride), bias=False),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.extract(x)
        return x

class Conv1Relu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Conv3Relu, self).__init__()
        self.extract = nn.Sequential(nn.Conv2d(in_ch, out_ch, (1, 1), padding=(1, 1),
                                               stride=(stride, stride), bias=False),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.extract(x)
        return x

class DropBlock(nn.Module):
    """
    [Ghiasi et al., 2018] DropBlock: A regularization method for convolutional networks
    """
    def __init__(self, rate=0.15, size=7, step=50):
        super().__init__()

        self.drop = LinearScheduler(
            DropBlock2D(block_size=size, drop_prob=0.),
            start_value=0,
            stop_value=rate,
            nr_steps=step
        )

    def forward(self, feats: list):
        if self.training:  # 只在训练的时候加上dropblock
            for i, feat in enumerate(feats):
                feat = self.drop(feat)
                feats[i] = feat
        return feats

    def step(self):
        self.drop.step()
     
class MSFI(nn.Module):
    def __init__(self, inplanes, neck_name='fpn+ppm+fuse'):
        super().__init__()

        self.stage1_Conv1 = Conv3Relu(inplanes * 2, inplanes)  # channel: 2*inplanes ---> inplanes
        self.stage2_Conv1 = Conv3Relu(inplanes * 4, inplanes * 2)  # channel: 4*inplanes ---> 2*inplanes
        self.stage3_Conv1 = Conv3Relu(inplanes * 8, inplanes * 4)  # channel: 8*inplanes ---> 4*inplanes
        self.stage4_Conv1 = Conv3Relu(inplanes * 16, inplanes * 8)  # channel: 16*inplanes ---> 8*inplanes

        self.stage2_Conv_after_up = Conv3Relu(inplanes * 2, inplanes)
        self.stage3_Conv_after_up = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage4_Conv_after_up = Conv3Relu(inplanes * 8, inplanes * 4)

        self.stage1_Conv2 = Conv3Relu(inplanes * 2, inplanes)
        self.stage2_Conv2 = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage3_Conv2 = Conv3Relu(inplanes * 8, inplanes * 4)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # 高宽扩大2倍
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True) # 高宽扩大4倍
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True) # 高宽扩大8倍
        self.d2 = Conv3Relu(inplanes * 2, inplanes)
        self.d3 = Conv3Relu(inplanes * 4, inplanes)
        self.d4 = Conv3Relu(inplanes * 8, inplanes)

        rate, size, step = (0.15, 7, 30)
        self.drop = DropBlock(rate=rate, size=size, step=step)


    def forward(self, ms_feats):
        fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4 = ms_feats
        change1_h, change1_w = fa1.size(2), fa1.size(3)

        [fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4] = self.drop([fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4])  # dropblock

        change1 = self.stage1_Conv1(torch.cat([fa1, fb1], 1))  # inplanes      --> 4, 64, 112, 112
        change2 = self.stage2_Conv1(torch.cat([fa2, fb2], 1))  # inplanes * 2  --> 4, 128, 56, 56
        change3 = self.stage3_Conv1(torch.cat([fa3, fb3], 1))  # inplanes * 4  --> 4, 256, 28, 28
        change4 = self.stage4_Conv1(torch.cat([fa4, fb4], 1))  # inplanes * 8  --> 4, 512, 14, 14
        # 以上形状未变
        


        change3_2 = self.stage4_Conv_after_up(self.up(change4)) # 4, 512, 14, 14 --> 4, 512, 28, 28 --> 4, 256, 28, 28

        change3 = self.stage3_Conv2(torch.cat([change3, change3_2], 1)) # 4, 512, 28, 28 --> 4, 256, 28, 28

        change2_2 = self.stage3_Conv_after_up(self.up(change3)) # 4, 256, 28, 28 --> 4, 256, 56, 56 --> 4, 128, 56, 56
        change2 = self.stage2_Conv2(torch.cat([change2, change2_2], 1)) # 4, 256, 56, 56 --> 4, 128, 56, 56

        change1_2 = self.stage2_Conv_after_up(self.up(change2)) # 4, 128, 56, 56 --> 4, 128, 112, 112 --> 4, 64, 112, 112
        change1 = self.stage1_Conv2(torch.cat([change1, change1_2], 1)) # 4, 128, 112, 112 --> 4, 64, 112, 112

        change = change1 # 4, 64, 112, 112
 
        return change4, change
class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        inter_channels = in_channels // 4
        self.head = nn.Sequential(Conv3Relu(in_channels, inter_channels),
                                  nn.Dropout(0.2),  # 使用0.1的dropout
                                  nn.Conv2d(inter_channels, out_channels, (1, 1)))
    def forward(self, x):
        return self.head(x)


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        #print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):#通道注意力加空间注意力模块
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        #print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out

class GatedFusionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(GatedFusionModule, self).__init__()
        hidden_dim = max(in_channels // reduction_ratio, 4)  # 确保隐藏层维度不为0
        self.CBAM = CBAM(in_channels)
        
        # 门控网络：学习如何加权融合不同特征
        self.gate_net = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_dim, 1),  # 输入是两个特征的拼接
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2, 3, padding=1),  # 输出2个通道的权重图
            nn.Softmax(dim=1)  # 保证权重和为1
        )
        
        # 可选的后处理卷积
        self.post_atten = nn.Sequential(
            nn.Conv2d(in_channels*2, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, feat_a, feat_b):
        """
        参数:
            feat_a: 第一个特征图 [B, C, H, W] (如差分特征)
            feat_b: 第二个特征图 [B, C, H, W] (如融合特征)
        返回:
            融合后的特征 [B, C, H, W]
        """
        feat_a, feat_b = self.CBAM(feat_a), self.CBAM(feat_b)
        # 拼接输入特征
        fused_cat = torch.cat([feat_a, feat_b], dim=1)  # [B, 2C, H, W]

        fused_1 = self.post_atten(fused_cat) * self.decode(fused_cat)
        
        # 生成空间自适应权重图
        gate_weights = self.gate_net(fused_cat)  # [B, 2, H, W]
        
        # 拆分为对应两个特征的权重
        w_a, w_b = gate_weights.chunk(2, dim=1)  # 各为[B, 1, H, W]
        
        # 加权融合
        fused_2 = feat_a * w_a + feat_b * w_b
        fused = fused_2 + fused_1
        
        return fused

class BidirectionalCrossAttention_window_gate(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=8, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.gated_fusion = GatedFusionModule(in_channels=dim)
        
        self.q1 = nn.Conv2d(dim, dim, 1)
        self.k1 = nn.Conv2d(dim, dim, 1)
        self.v1 = nn.Conv2d(dim, dim, 1)
        self.q2 = nn.Conv2d(dim, dim, 1)
        self.k2 = nn.Conv2d(dim, dim, 1)
        self.v2 = nn.Conv2d(dim, dim, 1)

        # self.proj = nn.Conv2d(dim * 2, dim, 1)
        self.norm = nn.BatchNorm2d(dim)
        self.dropout = nn.Dropout(dropout)

    def window_partition(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        x = x.view(B, C, H // ws, ws, W // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, nH, nW, ws, ws, C]
        return x.view(-1, ws * ws, C)  # [B*nH*nW, ws*ws, C]

    def window_reverse(self, x, H, W):
        ws = self.window_size
        B = x.shape[0] // ((H // ws) * (W // ws))
        x = x.view(B, H // ws, W // ws, ws, ws, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(B, -1, H, W)

    def forward(self, out1, out3):
        B, C, H, W = out1.shape
        ws = self.window_size

        # Project QKV
        Q1 = self.q1(out1)
        K1 = self.k1(out3)
        V1 = self.v1(out3)
        
        Q2 = self.q2(out3)
        K2 = self.k2(out1)
        V2 = self.v2(out1)

        # 分窗口
        Q1 = self.window_partition(Q1)
        K1 = self.window_partition(K1)
        V1 = self.window_partition(V1)
        Q2 = self.window_partition(Q2)
        K2 = self.window_partition(K2)
        V2 = self.window_partition(V2)

        # 方向1: out1->out3
        attn1 = (Q1 @ K1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        out_a = attn1 @ V1

        # 方向2: out3->out1
        attn2 = (Q2 @ K2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        out_b = attn2 @ V2

        # 恢复窗口
        out_a = self.window_reverse(out_a, H, W)
        out_b = self.window_reverse(out_b, H, W)

        # 融合
        # fused = torch.cat([out_a, out_b], dim=1)
        # fused = self.proj(fused)
        fused = self.gated_fusion(out_a, out_b)
        fused = self.norm(fused)
        fused = F.relu(fused)
        fused = self.dropout(fused)
        return fused

class ChangeDiff(nn.Module):
    def __init__(self, inplanes):
        super().__init__()

        self.stage2_Conv_after_up = Conv3Relu(inplanes * 2, inplanes)
        self.stage3_Conv_after_up = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage4_Conv_after_up = Conv3Relu(inplanes * 8, inplanes * 4)

        self.stage1_Conv2 = Conv3Relu(inplanes * 2, inplanes)
        self.stage2_Conv2 = Conv3Relu(inplanes * 4, inplanes * 2)
        self.stage3_Conv2 = Conv3Relu(inplanes * 8, inplanes * 4)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # 高宽扩大2倍
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True) # 高宽扩大4倍
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True) # 高宽扩大8倍
        self.d2 = Conv3Relu(inplanes * 2, inplanes)
        self.d3 = Conv3Relu(inplanes * 4, inplanes)
        self.d4 = Conv3Relu(inplanes * 8, inplanes)

        rate, size, step = (0.15, 7, 30)
        self.drop = DropBlock(rate=rate, size=size, step=step)


    def forward(self, ms_feats):
        fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4 = ms_feats
        change1_h, change1_w = fa1.size(2), fa1.size(3)

        [fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4] = self.drop([fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4])  # dropblock
        change1 = torch.abs(fa1 - fb1)
        change2 = torch.abs(fa2 - fb2)
        change3 = torch.abs(fa3 - fb3)
        change4 = torch.abs(fa4 - fb4)

        change3_2 = self.stage4_Conv_after_up(self.up(change4)) # 4, 512, 14, 14 --> 4, 512, 28, 28 --> 4, 256, 28, 28

        change3 = self.stage3_Conv2(torch.cat([change3, change3_2], 1)) # 4, 512, 28, 28 --> 4, 256, 28, 28

        change2_2 = self.stage3_Conv_after_up(self.up(change3)) # 4, 256, 28, 28 --> 4, 256, 56, 56 --> 4, 128, 56, 56
        change2 = self.stage2_Conv2(torch.cat([change2, change2_2], 1)) # 4, 256, 56, 56 --> 4, 128, 56, 56

        change1_2 = self.stage2_Conv_after_up(self.up(change2)) # 4, 128, 56, 56 --> 4, 128, 112, 112 --> 4, 64, 112, 112
        change1 = self.stage1_Conv2(torch.cat([change1, change1_2], 1)) # 4, 128, 112, 112 --> 4, 64, 112, 112

        change = change1 # 4, 64, 112, 112

        return change

            
class WaveDSNet(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.inplanes = int(opt.backbone.split("_")[-1]) if opt.backbone.startswith('cswin') else 64
        self.scale = ScaleInOutput(opt.input_size)
        self._create_backbone(opt.backbone, opt)
        print(f'==>use {opt.neck} for neck')
        self.neck = MSFI(self.inplanes)

        self.changediff = ChangeDiff(self.inplanes)

        # CSDI分支
        self.bca = BidirectionalCrossAttention_window_gate(dim=self.inplanes, num_heads=4, dropout=0.1)

        # BASE双分支
        self.final_seg_head = nn.Sequential(
                Conv3Relu(self.inplanes, 16),  # fusion_dim 是融合后的总通道数
                nn.Dropout(0.2),
                nn.Conv2d(16, 2, (1, 1)))
        self.edge_head = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 如果需要进一步放大输出
                    Conv3Relu(self.inplanes, 16),  # fusion_dim 是融合后的总通道数
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 如果需要进一步放大输出
                    nn.Dropout(0.2),
                    nn.Conv2d(16, 2, (1, 1)))

    def forward(self, xa, xb, names=None):
        # print(names)
        xa, xb = self.scale.scale_input((xa, xb))
        _, _, h_input, w_input = xa.shape
        assert xa.shape == xb.shape, "The two images are not the same size, please check it."
        out_size = h_input, w_input

        fa1, fa2, fa3, fa4 = self.backbone(xa) 
        fb1, fb2, fb3, fb4 = self.backbone(xb)
        
        ms_feats = fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4  
        change4, change = self.neck(ms_feats)
        
        out1 = change
        out2 = self.changediff(ms_feats)
        out4 = self.bca(out1, out2)
        out5 = F.interpolate(self.final_seg_head(out4), size=(h_input, w_input), mode='bilinear', align_corners=True)
        out6 = self.edge_head(out4)
        
        block = self.scale.scale_output(out5)[0]
        edge = self.scale.scale_output(out6)[0]

        return block, edge



    def _init_weight(self, pretrain=''):  
        for m in self.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):  
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if pretrain.endswith('.pth'):
            pretrained_dict = {k.replace('module.',''): v for k, v in torch.load(pretrain).items()}
            if isinstance(pretrained_dict, nn.DataParallel):
                pretrained_dict = pretrained_dict.module
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(OrderedDict(model_dict), strict=False)
            print("=> ChangeDetection load {}/{} items from: {}".format(len(pretrained_dict),
                                                                        len(model_dict), pretrain))
            # input()
    def _create_backbone(self, backbone, opt):
        if 'cswin' in backbone:
            if '_t_' in backbone:
                self.backbone = CSWin_64_12211_tiny_224(pretrained=True, opt=opt)
            elif '_s_' in backbone:
                self.backbone = CSWin_64_24322_small_224(pretrained=True, opt=opt)
            elif '_b_' in backbone:
                self.backbone = CSWin_96_24322_base_224(pretrained=True, opt=opt)
            elif '_b448_' in backbone:
                self.backbone = CSWin_96_24322_base_224(pretrained=True, opt=opt)
        else:
            raise Exception('Not Implemented yet: {}'.format(backbone))