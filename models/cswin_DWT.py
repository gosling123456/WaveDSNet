# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'cswin_224': _cfg(),
    'cswin_384': _cfg(
        crop_pct=1.0
    ),

}



class WTConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 wavelet='haar', levels=1, mode='zero', kernel_num=4, param_reduction=1.0):
        super(WTConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.wavelet = wavelet
        self.levels = levels
        self.kernel_num = kernel_num
        self.param_reduction = param_reduction

        # 小波变换模块
        self.dwt = DWTForward(J=levels, wave=wavelet, mode=mode)
        self.idwt = DWTInverse(wave=wavelet, mode=mode)
        
        
        # 初始化小波参数
        self.init_wavelet_parameters()
        
        # 注意力相关的投影层
        self.attention_proj = nn.Conv2d(in_channels, kernel_num, 1)
        
    def init_wavelet_parameters(self):
        """初始化小波域参数"""
        # 计算总子带数：1个低频 + 3*levels个高频
        self.total_subbands = 1 + 3 * self.levels
        
        if self.param_reduction < 1:
            self.total_subbands = int(self.total_subbands * self.param_reduction)
        
        # 小波核权重 [kernel_num, subbands, out_ch, in_ch, k, k]
        self.wt_kernels = nn.Parameter(
            torch.randn(self.kernel_num, self.total_subbands, 
                       self.out_channels, self.in_channels, 
                       self.kernel_size, self.kernel_size) * 1e-6
        )
        
    def generate_dynamic_kernel(self, subband_feat, attention_weights, subband_index): # yl, None, 0
        """为特定小波子带生成动态卷积核"""
        b, c, h, w = subband_feat.shape
        
        # 计算子带特定的注意力
        subband_att = self.attention_proj(subband_feat)  # [b, kernel_num, h, w]
        subband_att = F.adaptive_avg_pool2d(subband_att, (1, 1))  # [b, kernel_num, 1, 1]
        subband_att = F.softmax(subband_att, dim=1)  # 归一化注意力
        
        # 加权融合基础核
        kernel_weights = self.wt_kernels[:, subband_index]  # [kernel_num, out_ch, in_ch, k, k]
        weighted_kernels = (kernel_weights * subband_att.view(b, self.kernel_num, 1, 1, 1, 1)).sum(dim=1)
        # weighted_kernels形状: [b, out_ch, in_ch, k, k]
        
        return weighted_kernels
    
    def apply_wavelet_convolution(self, subband_feat, dynamic_kernel):
        """在小波子带上应用动态卷积"""
        b, c, h, w = subband_feat.shape
        
        # 对每个样本应用专属卷积核
        outputs = []
        for i in range(b):
            # 重塑卷积核为[out_ch, in_ch, k, k]
            sample_kernel = dynamic_kernel[i]
            # 应用卷积
            conv_output = F.conv2d(subband_feat[i:i+1], sample_kernel, 
                                 stride=self.stride, padding=self.padding)
            outputs.append(conv_output)
        
        return torch.cat(outputs, dim=0)
    
    def combine_wavelet_outputs(self, wavelet_outputs, yl_low_freq, yh_high_freq):
        """融合小波域卷积结果并进行逆变换"""
        reconstructed_yl = wavelet_outputs[0]  # 低频分量
        
        reconstructed_yh = []
        for i in range(self.levels):
            level_hf = []
            for j in range(3):  # 三个高频方向
                hf_idx = 1 + i * 3 + j
                if hf_idx < len(wavelet_outputs):
                    level_hf.append(wavelet_outputs[hf_idx].unsqueeze(2))
            
            if level_hf:
                reconstructed_yh.append(torch.cat(level_hf, dim=2))
        
        # 逆小波变换重构
        if reconstructed_yh:
            final_output = self.idwt((reconstructed_yl, reconstructed_yh))
        else:
            final_output = reconstructed_yl
            
        return final_output
    
    def forward(self, x):
        b, c, h, w = x.shape
        # 1. 小波分解
        yl, yh = self.dwt(x)
        
        # 2. 为每个小波子带生成动态卷积核并应用卷积
        wavelet_results = []
        
        # 处理低频分量
        ll_kernel = self.generate_dynamic_kernel(yl, None, 0)
        ll_conv = self.apply_wavelet_convolution(yl, ll_kernel)
        wavelet_results.append(ll_conv)
        # input('这里看下前面的结果')
        # 处理高频分量
        for level in range(min(self.levels, len(yh))):
            level_hf = yh[level]  # [b, c, 3, h_l, w_l]
            
            for dir_idx in range(3):
                subband_idx = 1 + level * 3 + dir_idx
                if subband_idx >= self.total_subbands:
                    continue
                    
                # 提取特定方向的高频特征
                dir_feat = level_hf[:, :, dir_idx, :, :]  # [b, c, h_l, w_l]
                
                # 生成动态核并卷积
                dir_kernel = self.generate_dynamic_kernel(dir_feat, None, subband_idx)
                dir_conv = self.apply_wavelet_convolution(dir_feat, dir_kernel)
                wavelet_results.append(dir_conv)
        
        # 3. 融合所有小波域卷积结果
        output = self.combine_wavelet_outputs(wavelet_results, yl, yh)
        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp) ### B', C, H', W'

        lepe = func(x) ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp* self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q,k,v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        
        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x


class CSWinBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        
        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx = -1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim//2, resolution=self.patches_resolution, idx = i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        
        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:,:,:,:C//2])
            x2 = self.attns[1](qkv[:,:,:,C//2:])
            attened_x = torch.cat([x1,x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm, opt=None):
        super().__init__()
        self.conv = WTConv(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        
        return x


class CSWinTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=96, depth=[2,2,6,2], split_size = [3,5,7],
                 num_heads=(2,4,8,16), mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False, opt=None):
        super().__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads = num_heads

        self.stage1_conv_embed = nn.Sequential(
            WTConv(in_chans, embed_dim, 7, 4, 2),
            # nn.Conv2d(in_chans, embed_dim, 7, 4, 2),
            Rearrange('b c h w -> b (h w) c', h=img_size//4, w=img_size//4),
            nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[0], reso=img_size//4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])

        self.merge1 = Merge_Block(curr_dim, curr_dim*2, opt=opt)
        curr_dim = curr_dim*2
        self.stage2 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size//8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1])+i], norm_layer=norm_layer)
            for i in range(depth[1])])
        
        self.merge2 = Merge_Block(curr_dim, curr_dim*2, opt=opt)
        curr_dim = curr_dim*2
        temp_stage3 = []
        temp_stage3.extend(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], reso=img_size//16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2])+i], norm_layer=norm_layer)
            for i in range(depth[2])])

        self.stage3 = nn.ModuleList(temp_stage3)
        
        self.merge3 = Merge_Block(curr_dim, curr_dim*2, opt=opt)
        curr_dim = curr_dim*2
        self.stage4 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[3], reso=img_size//32, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1])+i], norm_layer=norm_layer, last_stage=True)
            for i in range(depth[-1])])
       
        self.norm = norm_layer(curr_dim)
        # Classifier head
        self.head = nn.Linear(curr_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def reset_classifier(self, num_classes, global_pool=''):
        if self.num_classes != num_classes:
            print('reset head to', num_classes)
            self.num_classes = num_classes
            self.head = nn.Linear(self.out_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.head = self.head.cuda()
            trunc_normal_(self.head.weight, std=.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def forward_features(self, x):
        feature1, feature2, feature3, feature4 = None, None, None, None
        B = x.shape[0]
        x = self.stage1_conv_embed(x)
        for blk in self.stage1:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        feature1 = x
        for i, (pre, blocks) in enumerate(zip([self.merge1, self.merge2, self.merge3],
                                        [self.stage2, self.stage3, self.stage4])):
            x = pre(x)
            for blk in blocks:
                if self.use_chk:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
            if i == 0:
                feature2 = x
            elif i == 1:
                feature3 = x
            elif i == 2:
                feature4 = x

        # x = self.norm(x)
        # return torch.mean(x, dim=1)

        feature1 = feature1.view(feature1.shape[0], int(np.sqrt(feature1.shape[1])),
                                 int(np.sqrt(feature1.shape[1])), feature1.shape[2])
        feature2 = feature2.view(feature2.shape[0], int(np.sqrt(feature2.shape[1])),
                                 int(np.sqrt(feature2.shape[1])), feature2.shape[2])
        feature3 = feature3.view(feature3.shape[0], int(np.sqrt(feature3.shape[1])),
                                 int(np.sqrt(feature3.shape[1])), feature3.shape[2])
        feature4 = feature4.view(feature4.shape[0], int(np.sqrt(feature4.shape[1])),
                                 int(np.sqrt(feature4.shape[1])), feature4.shape[2])
        feature1 = feature1.permute(0, 3, 1, 2)
        feature2 = feature2.permute(0, 3, 1, 2)
        feature3 = feature3.permute(0, 3, 1, 2)
        feature4 = feature4.permute(0, 3, 1, 2)

        # print(feature1.shape)
        # print(feature2.shape)
        # print(feature3.shape)
        # print(feature4.shape)
        return feature1, feature2, feature3, feature4

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


# 224 models


@register_model
def CSWin_64_12211_tiny_224(pretrained=False, **kwargs):
    model = CSWinTransformer(img_size=448, patch_size=4, embed_dim=64, depth=[1, 2, 21, 1],
                             split_size=[1, 2, 7, 7], num_heads=[2, 4, 8, 16], mlp_ratio=4., **kwargs)
    # model = CSWinTransformer(img_size=448, patch_size=4, embed_dim=64, depth=[1, 2, 21, 1], split_size=[1, 2, 7, 7],
    #                          num_heads=[2, 4, 8, 16], mlp_ratio=4., drop_rate=0.2, drop_path_rate=0.2, **kwargs)
    if pretrained:
        pretrained_file = "./models/cswin_tiny_224.pth"
        pretrained_dict = torch.load(pretrained_file, map_location='cpu')["state_dict_ema"]
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(OrderedDict(model_dict), strict=False)
        print('\n=> load {}/{} items for CSwin-T from pretrained model: {}'.
              format(len(pretrained_dict), len(model_dict), pretrained_file))
    return model


@register_model
def CSWin_64_24322_small_224(pretrained=False, **kwargs):
    model = CSWinTransformer(img_size=448, patch_size=4, embed_dim=64, depth=[2, 4, 32, 2],
                             split_size=[1, 2, 7, 7], num_heads=[2, 4, 8, 16], mlp_ratio=4., **kwargs)
    if pretrained:
        pretrained_file = "./models/cswin_small_224.pth"
        pretrained_dict = torch.load(pretrained_file)["state_dict_ema"]
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(OrderedDict(model_dict), strict=False)
        print('\n=> load {}/{} items for CSwin-S from pretrained model: {}'.
              format(len(pretrained_dict), len(model_dict), pretrained_file))
    return model


@register_model
def CSWin_96_24322_base_224(pretrained=False, **kwargs):
    model = CSWinTransformer(img_size=448, patch_size=4, embed_dim=96, depth=[2, 4, 32, 2],
                             split_size=[1, 2, 7, 7], num_heads=[4, 8, 16, 32], mlp_ratio=4., **kwargs)
    if pretrained:
        pretrained_file = "./models/cswin_base_224.pth"
        pretrained_dict = torch.load(pretrained_file)["state_dict_ema"]
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(OrderedDict(model_dict), strict=False)
        print('\n=> load {}/{} items for CSwin-B-224 from pretrained model: {}'.
              format(len(pretrained_dict), len(model_dict), pretrained_file))
    return model


@register_model
def CSWin_144_24322_large_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2,4,32,2],
        split_size=[1,2,7,7], num_heads=[6,12,24,24], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model


### 384 models


@register_model
def CSWin_96_24322_base_384(pretrained=False, **kwargs):
    model = CSWinTransformer(img_size=384, patch_size=4, embed_dim=96, depth=[2, 4, 32, 2],
                             split_size=[1, 2, 12, 12], num_heads=[4, 8, 16, 32], mlp_ratio=4., **kwargs)
    if pretrained:
        pretrained_file = "./models/cswin_base_384.pth"
        pretrained_dict = torch.load(pretrained_file)["state_dict_ema"]
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(OrderedDict(model_dict), strict=False)
        print('\n=> load {}/{} items for CSwin-B from pretrained model: {}'.
              format(len(pretrained_dict), len(model_dict), pretrained_file))
    return model


@register_model
def CSWin_144_24322_large_384(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2,4,32,2],
        split_size=[1,2,12,12], num_heads=[6,12,24,24], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_384']
    return model