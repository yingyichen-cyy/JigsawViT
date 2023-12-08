# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import einops

from timm.models.vision_transformer import PatchEmbed, Block

import utils

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self,
                 nb_cls=10,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = torch.nn.Linear(embed_dim, nb_cls)
        self.jigsaw = torch.nn.Sequential(*[torch.nn.Linear(embed_dim, embed_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(embed_dim, embed_dim),
                                          torch.nn.ReLU(),
                                          torch.nn.Linear(embed_dim, self.num_patches)])
        self.target = torch.arange(self.num_patches)
        

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = utils.get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # target = einops.repeat(self.target, 'L -> N L', N=N) 
        # target = target.to(x.device)
        
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep] # N, len_keep
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        target_masked = ids_keep
        
        return x_masked, target_masked

    def forward_jigsaw(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # masking: length -> length * mask_ratio
        x, target = self.random_masking(x, mask_ratio)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.jigsaw(x[:, 1:])
        return x.reshape(-1, self.num_patches), target.reshape(-1)
    
    def forward_cls(self, x) : 
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x

    def forward(self, x_jigsaw, x_cls, mask_ratio) :
        pred_jigsaw, targets_jigsaw = self.forward_jigsaw(x_jigsaw, mask_ratio)
        pred_cls = self.forward_cls(x_cls)
        return pred_jigsaw, targets_jigsaw, pred_cls
        

def mae_vit_small_patch16(nb_cls, **kwargs):
    model = MaskedAutoencoderViT(nb_cls,
                                 img_size=224,
                                 patch_size=16,
                                 embed_dim=384,
                                 depth=12,
                                 num_heads=6,
                                 mlp_ratio=4,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 **kwargs)
    return model


def mae_vit_base_patch16(nb_cls, **kwargs):
    model = MaskedAutoencoderViT(nb_cls,
                                 img_size=224,
                                 patch_size=16,
                                 embed_dim=768,
                                 depth=12,
                                 num_heads=12,
                                 mlp_ratio=4,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 **kwargs)
    return model


def mae_vit_large_patch16(nb_cls, **kwargs):
    model = MaskedAutoencoderViT(nb_cls,
                                 img_size=224,
                                 patch_size=16,
                                 embed_dim=1024,
                                 depth=24,
                                 num_heads=16,
                                 mlp_ratio=4,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                 **kwargs)
    return model


def create_model(arch, nb_cls) :     
    if arch == 'vit_small_patch16' : 
        return mae_vit_small_patch16(nb_cls)
    elif arch == 'vit_base_patch16' : 
        return mae_vit_base_patch16(nb_cls)
    elif arch == 'vit_large_patch16' : 
        return mae_vit_large_patch16(nb_cls)
    
if __name__ == '__main__':
    
    net = create_model(arch = 'vit_small_patch16', nb_cls = 10) 
    net = net.cuda()
    img = torch.cuda.FloatTensor(6, 3, 224, 224)
    mask_ratio = 0.75
    with torch.no_grad():
        x, target = net.forward_jigsaw(img, mask_ratio)