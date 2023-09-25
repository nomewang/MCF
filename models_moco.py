import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import os

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        out_list = [2,5,8,11]

        out_feat = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in out_list:
                out_feat.append(x)


        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome, out_feat


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

class CPLoss(nn.Module):
    def __init__(self, path):
        super(CPLoss, self).__init__()
        print("Loading MAE model from path: {}".format(path))
        self.model = self.__load_model(path)
        self.model.cuda()
        self.model.eval()

    @staticmethod
    def __load_model(path):
        model = vit_base_patch16(num_classes=1000, global_pool=False)

        print("Load pre-trained checkpoint mocov3 from: %s" % path)
        linear_keyword = 'head'
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

            print("=> loaded pre-trained model '{}'".format(path))
        else:
            print("=> no checkpoint found at '{}'".format(path))


        for _, p in model.named_parameters():
            p.requires_grad = False

        return model

    def extract_feats(self, x):
        x = F.interpolate(x, size=224)
        _ , x_feats  = self.model.forward_features(x)

        x_feats_norm = []
        for feat in x_feats:
            feat = nn.functional.normalize(feat, dim=1)
            x_feats_norm.append(feat)
        return x_feats_norm

    def forward(self, y_hat, y, norm=True, mask=None):
        if norm:
            y_hat = TF.resize(y_hat, y.shape[-1])
            y_hat = TF.normalize(y_hat, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        y_feats = self.extract_feats(y)
        y_hat_feats = self.extract_feats(y_hat)
        loss = 0
        for y_hat_feat, y_feat in zip(y_hat_feats, y_feats):            
            loss += F.mse_loss(y_hat_feat, y_feat.detach())
        return loss
