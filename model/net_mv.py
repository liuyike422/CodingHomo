import os
import sys
o_path = os.getcwd()
o_path = o_path + '/model'
print(o_path)
sys.path.append(o_path)
import math

"""Defines the neural network, losss function and metrics"""
import cv2
import torch
import imageio
import torch.nn as nn
from torchvision.utils import save_image
import torch.utils.model_zoo as model_zoo

from swin_multi import *
from timm.models.layers import trunc_normal_


from torchvision import utils
import torch.nn.functional as F
from model.module.aspp import ASPP
from model.nn_upsample import NeuralUpsampler, upsample2d_flow_as
from model.utils import get_warp_flow, upsample2d_flow_as, get_grid, homo_flow_gen


__all__ = ['HomoNet', 'Discriminator']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def make_gif(img1, img2, name):
    img1, img2 = cv2.imread(img1), cv2.imread(img2)
    
    with imageio.get_writer(name+'.gif', mode='I', duration = 0.5,loop = 0) as writer:
    
        writer.append_data(img1.astype(np.uint8))
    
        writer.append_data(img2.astype(np.uint8))


def tensor_erode(bin_img, ksize=5):
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)

    eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
    return eroded


def tensor_dilation(bin_img, ksize=5):
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)

    dilation, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)
    return dilation


def gen_basis(h, w, is_qr=True, is_scale=True):
    basis_nb = 8
    grid = get_grid(1, h, w).permute(0, 2, 3, 1)  # 1, w, h, (x, y, 1)
    flow = grid[:, :, :, :2] * 0

    names = globals()
    for i in range(1, basis_nb + 1):
        names['basis_' + str(i)] = flow.clone()

    basis_1[:, :, :, 0] += grid[:, :, :, 0]  # [1, w, h, (x, 0)]
    basis_2[:, :, :, 0] += grid[:, :, :, 1]  # [1, w, h, (y, 0)]
    basis_3[:, :, :, 0] += 1  # [1, w, h, (1, 0)]
    basis_4[:, :, :, 1] += grid[:, :, :, 0]  # [1, w, h, (0, x)]
    basis_5[:, :, :, 1] += grid[:, :, :, 1]  # [1, w, h, (0, y)]
    basis_6[:, :, :, 1] += 1  # [1, w, h, (0, 1)]
    basis_7[:, :, :, 0] += grid[:, :, :, 0] ** 2  # [1, w, h, (x^2, xy)]
    basis_7[:, :, :, 1] += grid[:, :, :, 0] * grid[:, :, :, 1]  # [1, w, h, (x^2, xy)]
    basis_8[:, :, :, 0] += grid[:, :, :, 0] * grid[:, :, :, 1]  # [1, w, h, (xy, y^2)]
    basis_8[:, :, :, 1] += grid[:, :, :, 1] ** 2  # [1, w, h, (xy, y^2)]

    flows = torch.cat([names['basis_' + str(i)] for i in range(1, basis_nb + 1)], dim=0)
    if is_qr:
        flows_ = flows.view(basis_nb, -1).permute(1, 0)  # N, h, w, c --> N, h*w*c --> h*w*c, N
        flow_q, _ = torch.qr(flows_)
        flow_q = flow_q.permute(1, 0).reshape(basis_nb, h, w, 2)
        flows = flow_q

    if is_scale:
        max_value = flows.abs().reshape(8, -1).max(1)[0].reshape(8, 1, 1, 1)
        flows = flows / max_value

    
    return flows.permute(0, 3, 1, 2)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, n_classes=1):
        super(Discriminator, self).__init__()
        self.cls_head = self.cls_net(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv_last = nn.Conv2d(512, n_classes, kernel_size=1, padding=0, stride=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def cls_net(input_channels, kernel_size=3, padding=1):
        layers = []
        channels = [input_channels * 2, 32, 64, 128, 256, 512]
        for i in range(len(channels) - 1):
            layers.append(
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size, padding=padding, stride=2, bias=False))
            layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.cls_head(x)
        bs = len(x)
        x = self.conv_last(x)
        x = self.pool(x).view(bs, -1)
        return x


class HomoNet(nn.Module):
    # 224*224
    def __init__(self, params, backbone, init_mode="resnet", norm_layer=nn.LayerNorm):
        super(HomoNet, self).__init__()

        self.init_mode = init_mode
        self.params = params
        self.fea_extra = self.feature_extractor(self.params.in_channels, 1)
        self.h_net = backbone(params, norm_layer=norm_layer)
        self.basis = gen_basis(self.params.crop_size[0], self.params.crop_size[1]).unsqueeze(0).reshape(1, 8, -1)
        self.apply(self._init_weights)
        # self.mask_pred = self.mask_predictor(32)

        self.mask_predictor_fea = MaskEstimator(2, (8, 16, 32, 16, 8), 1)
        # self.mask_predictor_mv = MaskEstimator(4, (8, 16, 32, 16, 8), 1)
        self.mask_predictor = MaskEstimator(4, (8, 16, 32, 16, 8), 1)

    def _init_weights(self, m):
        if "swin" in self.init_mode:
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        elif "resnet" in self.init_mode:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    @staticmethod
    def feature_extractor(input_channels, out_channles, kernel_size=3, padding=1):
        layers = []
        channels = [input_channels // 2, 4, 8, out_channles]
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(channels[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    

    def forward(self, data_batch, step):
        img1_full, img2_full = data_batch["imgs_gray_full"][:, :1, :, :], data_batch["imgs_gray_full"][:, 1:, :, :]
        img1_patch, img2_patch = data_batch["imgs_gray_patch"][:, :1, :, :], data_batch["imgs_gray_patch"][:, 1:, :, :]
        bs, _, h_patch, w_patch = data_batch["imgs_gray_patch"].size()
        start, src_pt = data_batch['start'], data_batch['pts']

        #mv_flow
        mv_flow_b = data_batch["mv_flow_patch"]
        mv_flow_f = -get_warp_flow(mv_flow_b.clone(), mv_flow_b)

        # ==========================full features======================================
        img1_patch_fea, img2_patch_fea = list(map(self.fea_extra, [img1_patch, img2_patch]))
        img1_full_fea, img2_full_fea = list(map(self.fea_extra, [img1_full, img2_full]))

        img1_patch_fea = img1_patch_fea.detach()
        img2_patch_fea = img2_patch_fea.detach()
        img1_full_fea = img1_full_fea.detach()
        img2_full_fea = img2_full_fea.detach()

        # ========================forward ====================================

        forward_fea = torch.cat([img1_patch_fea, img2_patch_fea], dim=1)
        weight_f, pyramid_flow_f, pyramid_weight_f, mask_f = self.h_net(forward_fea, mv_flow_f)
        H_flow_f = (self.basis.to(forward_fea.device) * weight_f).sum(1).reshape(bs, 2, h_patch, w_patch)
        mask_f_mv = self.mask_predictor( torch.cat([mv_flow_f.detach(), H_flow_f.detach()], dim=1))
        mask_f.append(mask_f_mv)

        # ========================backward===================================

        backward_fea = torch.cat([img2_patch_fea, img1_patch_fea], dim=1)
        weight_b, pyramid_flow_b, pyramid_weight_b, mask_b = self.h_net(backward_fea, mv_flow_b)
        H_flow_b = (self.basis.to(backward_fea.device) * weight_b).sum(1).reshape(bs, 2, h_patch, w_patch)
        mask_b_mv = self.mask_predictor(torch.cat([mv_flow_b.detach(), H_flow_b.detach()], dim=1))
        mask_b.append(mask_b_mv)

        if self.training:
            warp_img1_patch, warp_img1_patch_fea = list(
                map(lambda x: get_warp_flow(x, H_flow_b, start), [img1_full, img1_full_fea]))
            warp_img2_patch, warp_img2_patch_fea = list(
                map(lambda x: get_warp_flow(x, H_flow_f, start), [img2_full, img2_full_fea]))

        else:
            warp_img1_patch, warp_img1_patch_fea = list(
                map(lambda x: get_warp_flow(x, H_flow_b, start), [img1_patch, img1_patch_fea]))
            warp_img2_patch, warp_img2_patch_fea = list(
                map(lambda x: get_warp_flow(x, H_flow_f, start), [img2_patch, img2_patch_fea]))


        # img1_patch_warp_fea = self.fea_extra(warp_img1_patch)
        img1_patch_warp_fea, img2_patch_warp_fea = list(
            map(self.fea_extra, [warp_img1_patch, warp_img2_patch]))
        
        img1_patch_warp_fea = img1_patch_warp_fea.detach()
        img2_patch_warp_fea = img2_patch_warp_fea.detach()

        # ============================= mask==========================================
        mask_f_pic = self.mask_predictor_fea(torch.cat((img1_patch_fea.detach(), warp_img2_patch_fea.detach()), dim=1))
        mask_b_pic = self.mask_predictor_fea(torch.cat((img2_patch_fea.detach(), warp_img1_patch_fea.detach()), dim=1))            
        mask_f_open = tensor_dilation(mask_f_pic)
        mask_f_open = tensor_erode(mask_f_open)
        mask_b_open = tensor_dilation(mask_b_pic)
        mask_b_open = tensor_erode(mask_b_open)

        mask_f.append(mask_f_open)
        mask_b.append(mask_b_open)


        mask_f_cross = mask_f[3] * mask_f[2]
        mask_f.append(mask_f_cross)
        mask_b_cross = mask_b[3] * mask_b[2]
        mask_b.append(mask_b_cross)

        if not self.training:
            H_flow_f = upsample2d_flow_as(H_flow_f, img1_full, "bilinear", True)
            H_flow_b = upsample2d_flow_as(H_flow_b, img1_full, "bilinear", True)
            for i in range(len(mask_b)):
                mask_b[i] = upsample2d_flow_as(mask_b[i], img1_full, "bilinear", False)
                mask_f[i] = upsample2d_flow_as(mask_f[i], img1_full, "bilinear", False)

            for i in range(len(pyramid_flow_b)):
                pyramid_flow_b[i] = upsample2d_flow_as(pyramid_flow_b[i], img1_full, "bilinear", True)#.permute(0, 2, 3, 1)
                pyramid_flow_f[i] = upsample2d_flow_as(pyramid_flow_f[i], img1_full, "bilinear", True)#.permute(0, 2, 3, 1)

        H_flow_f, H_flow_b = H_flow_f.permute(0, 2, 3, 1), H_flow_b.permute(0, 2, 3, 1)

        return {'img1_full_fea': img1_full_fea, 'img2_full_fea':img2_full_fea,
                'warp_img1_patch_fea': warp_img1_patch_fea, 'warp_img2_patch_fea': warp_img2_patch_fea,
                'img1_patch_warp_fea': img1_patch_warp_fea, 'img2_patch_warp_fea': img2_patch_warp_fea,
                'warp_img1_patch': warp_img1_patch, 'warp_img2_patch': warp_img2_patch,
                'img1_patch_fea': img1_patch_fea, 'img2_patch_fea': img2_patch_fea,
                'flow_f': H_flow_f, 'flow_b': H_flow_b,
                }


def Ms_Transformer(pretrained=False, **kwargs):
    """Constructs a Multi-scale Transformer model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = HomoNet(backbone=SwinTransformerMV, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def fetch_net(params):
    if params.net_type == "CodingHomo":
        HNet = Ms_Transformer(params=params)
    else:
        raise NotImplementedError
    return HNet
