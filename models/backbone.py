# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

# wsod
class DINOBackbone(nn.Module):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, args: str):
        super().__init__()
        import models.vision_transformer as vits
        self.arch = args.arch
        self.patch_size = args.patch_size
        # self.conv = nn.Conv2d(6, args.hidden_dim, 1)
        self.num_channels = args.hidden_dim

        self.model = vits.__dict__[self.arch](patch_size=self.patch_size, num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        
        url = self.return_url()
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        self.model.load_state_dict(state_dict, strict=True)

    def return_url(self):
        if self.arch == "vit_small" and self.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif self.arch == "vit_small" and self.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
        elif self.arch == "vit_base" and self.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif self.arch == "vit_base" and self.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"

        return url

    def forward(self, tensor_list: NestedTensor):
        w_featmap = tensor_list.tensors.shape[-2] // self.patch_size
        h_featmap = tensor_list.tensors.shape[-1] // self.patch_size
        
        # get_intermediate_layers로 중간 layer들 뽑아서 쓰는것도 가능은 할 것 같은데 
        attentions, _x_ctxed, _x_final = self.model.get_last_selfattention(tensor_list.tensors)
        # height x width = # of tokens
        # attentions : query @ key, [batch, # of heads,  # of tokens, # of tokens]
        # _x_ctxed : attentions @ value, means contextualized, [batch, # of tokens, dimension]
        # _x_final : mlp(attentions), attention block(attention + mlp)을 완전히 통과한 이후의 tokens 
        # -> 즉 frozen extractor에서 뽑은 feature를 쓰려면, _x_final을 써야함
        
        # nhead = attentions.shape[1] 
        # cls_attn = attentions.mean(1).squeeze()[0,1:].reshape(w_featmap, h_featmap)
        # cls_attn = attentions[:,:,0,1:].reshape(-1, nhead, w_featmap, h_featmap) # ex. 2, 6, 62, 75
        # cls_attn = self.conv(cls_attn) # 2, 256, w, h
        #cls_attn = self.conv(cls_attn).flatten(2).permute(1,0,1) # ex. torch.Size([2, 256, 9435]) > 4960, 2, 256
        # RuntimeError: Given groups=1, weight of size [256, 256, 1, 1], expected input[1, 9435, 2, 256] to have 256 channels, but got 9435 channels instead
        xs = {'last': _x_final} # _x_final : [batch, # of tokens, dimension], # of tokens = H_patches x W_patches
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)

        return out

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    # wsod
    if not args.backbone.startswith('resnet'): 
        backbone = DINOBackbone(args)
    else:
        train_backbone = args.lr_backbone > 0
        return_interm_layers = args.masks
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    
    model.num_channels = backbone.num_channels
    return model
