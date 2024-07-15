"""
PyTorch Implementation of Vision Transformer
("An Image is Worth 16X16 Words: Transformers for Image Recognition at Scale")

Reference
- Paper: https://arxiv.org/abs/2010.11929
- Code: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_1d.py
"""
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L137
class DropPath(nn.Module):
    '''
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    '''
    def __init__(self, drop_prob: float, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob <= 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class PreNorm(nn.Module):
    def __init__(self,
                 dim: int,
                 fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    MLP Module with GELU activation fn + dropout.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 drop_out_rate=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(drop_out_rate),
                                 nn.Linear(hidden_dim, output_dim),
                                 nn.Dropout(drop_out_rate))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 heads: int = 8,
                 dim_head: int = 64,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == input_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_drop_out_rate)
        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=qkv_bias)

        if project_out:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, output_dim),
                                        nn.Dropout(drop_out_rate))
        else:
            self.to_out = nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 heads: int = 8,
                 dim_head: int = 32,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.):
        super().__init__()
        attn = Attention(input_dim=input_dim,
                         output_dim=output_dim,
                         heads=heads,
                         dim_head=dim_head,
                         qkv_bias=qkv_bias,
                         drop_out_rate=drop_out_rate,
                         attn_drop_out_rate=attn_drop_out_rate)
        self.attn = PreNorm(dim=input_dim,
                            fn=attn)
        self.droppath1 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        ff = FeedForward(input_dim=output_dim,
                         output_dim=output_dim,
                         hidden_dim=hidden_dim,
                         drop_out_rate=drop_out_rate)
        self.ff = PreNorm(dim=output_dim,
                          fn=ff)
        self.droppath2 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.droppath1(self.attn(x)) + x
        x = self.droppath2(self.ff(x)) + x
        return x


class ViT(nn.Module):
    def __init__(self,
                 num_leads: int,
                 seq_len: int,
                 patch_size: int,
                 width: int = 768,
                 depth: int = 12,
                 mlp_dim: int = 3072,
                 heads: int = 12,
                 dim_head: int = 64,
                 qkv_bias: bool = True,
                 drop_out_rate: float = 0.,
                 attn_drop_out_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 **kwargs):
        super().__init__()
        assert seq_len % patch_size == 0, 'The sequence length must be divisible by the patch size.'
        num_patches = seq_len // patch_size
        
        # conv patch start
        self.to_patch_embedding = nn.Conv1d(num_leads, width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, width))

        self.dropout = nn.Dropout(drop_out_rate)

        
        self.depth = depth
        self.width = width
        drop_path_rate_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            block = TransformerBlock(input_dim=width,
                                     output_dim=width,
                                     hidden_dim=mlp_dim,
                                     heads=heads,
                                     dim_head=dim_head,
                                     qkv_bias=qkv_bias,
                                     drop_out_rate=drop_out_rate,
                                     attn_drop_out_rate=attn_drop_out_rate,
                                     drop_path_rate=drop_path_rate_list[i])
            self.add_module(f'block{i}', block)

        self.norm = nn.LayerNorm(width)
        self.head = nn.Identity()

    def forward_encoding(self, series):

        # for conv patch
        x = self.to_patch_embedding(series)
        x = rearrange(x, 'b c n -> b n c')
        x = x + self.pos_embedding

        # transformer blocks
        x = self.dropout(x)
        for i in range(self.depth):
            x = getattr(self, f'block{i}')(x)

        x = torch.mean(x, dim=1)  # global average pooling

        return self.norm(x)

    def forward(self, series):
        x = self.forward_encoding(series)
        x = self.head(x)
        return x

    def reset_head(self, num_classes=1):
        del self.head
        self.head = nn.Linear(self.width, num_classes)


def vit_tiny(num_leads, num_classes=1, seq_len=5000, patch_size=50, **kwargs):
    model_args = dict(num_leads=num_leads,
                      num_classes=num_classes,
                      seq_len=seq_len,
                      patch_size=patch_size,
                      width=192,
                      depth=12,
                      heads=3,
                      mlp_dim=768,
                      **kwargs)
    return ViT(**model_args)


def vit_small(num_leads, num_classes=1, seq_len=5000, patch_size=50, **kwargs):
    model_args = dict(num_leads=num_leads,
                      num_classes=num_classes,
                      seq_len=seq_len,
                      patch_size=patch_size,
                      width=384,
                      depth=12,
                      heads=6,
                      mlp_dim=1536,
                      **kwargs)
    return ViT(**model_args)


def vit_middle(num_leads, num_classes=1, seq_len=5000, patch_size=50, **kwargs):
    model_args = dict(num_leads=num_leads,
                      num_classes=num_classes,
                      seq_len=seq_len,
                      patch_size=patch_size,
                      width=512,
                      depth=12,
                      heads=8,
                      mlp_dim=2048,
                      **kwargs)
    return ViT(**model_args)


def vit_base(num_leads, num_classes=1, seq_len=5000, patch_size=50, **kwargs):
    model_args = dict(num_leads=num_leads,
                      num_classes=num_classes,
                      seq_len=seq_len,
                      patch_size=patch_size,
                      width=768,
                      depth=12,
                      heads=12,
                      mlp_dim=3072,
                      **kwargs)
    return ViT(**model_args)
