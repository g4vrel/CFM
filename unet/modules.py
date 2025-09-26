import math

import numpy as np

from einops import rearrange, parse_shape, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads=4, groups=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert in_channels % num_heads == 0, (
            "in_channels must be divisible by num_heads"
        )

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=in_channels)

        self.Q = nn.Conv2d(in_channels, in_channels, 1)
        self.K = nn.Conv2d(in_channels, in_channels, 1)
        self.V = nn.Conv2d(in_channels, in_channels, 1)

        self.proj = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)

        b, c, h_size, w_size = x.shape
        q = rearrange(
            self.Q(h),
            "b (head_dim num_heads) h w -> b num_heads (h w) head_dim",
            head_dim=self.head_dim,
            num_heads=self.num_heads,
        )
        k = rearrange(
            self.K(h),
            "b (head_dim num_heads) h w -> b num_heads (h w) head_dim",
            head_dim=self.head_dim,
            num_heads=self.num_heads,
        )
        v = rearrange(
            self.V(h),
            "b (head_dim num_heads) h w -> b num_heads (h w) head_dim",
            head_dim=self.head_dim,
            num_heads=self.num_heads,
        )

        out = F.scaled_dot_product_attention(q, k, v)

        out = rearrange(
            out,
            "b num_heads (h w) head_dim -> b (head_dim num_heads) h w",
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            h=h_size,
            w=w_size,
        )

        return (x + self.proj(out)) / np.sqrt(2.0)


def make_skip_connection(dim_in, dim_out):
    if dim_in == dim_out:
        return nn.Identity()
    return nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)


def make_attn(dim_out, attn):
    if not attn:
        return nn.Identity()
    return AttentionBlock(dim_out)


def make_block(dim_in, dim_out, num_groups, dropout=0):
    return nn.Sequential(
        nn.GroupNorm(num_groups=num_groups, num_channels=dim_in),
        nn.SiLU(),
        nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
        nn.Conv2d(dim_in, dim_out, 3, 1, 1),
    )


class ConditioningBlock(nn.Module):
    def __init__(self, dim_out, emb_dim, scale_shift=True):
        super().__init__()
        dim = 2 * dim_out if scale_shift else dim_out
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, dim))

    def forward(self, emb):
        emb = self.proj(emb)[:, :, None, None]
        return emb


class ResBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        emb_dim,
        scale_shift=True,
        num_groups=32,
        dropout=0.1,
        attn=False,
    ):
        super().__init__()
        self.scale_shift = scale_shift

        self.skip_connection = make_skip_connection(dim_in, dim_out)

        self.block1 = make_block(dim_in, dim_out, num_groups, dropout=0)
        self.block2 = make_block(dim_out, dim_out, num_groups, dropout=dropout)
        self.cond_block = ConditioningBlock(dim_out, emb_dim, scale_shift)
        self.attn = make_attn(dim_out, attn)

    def forward(self, x, emb):
        emb = self.cond_block(emb)

        h = self.block1(x)
        if self.scale_shift:
            out_norm, out_rest = self.block2[0], self.block2[1:]
            scale, shift = emb.chunk(2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb
            h = self.block2(h)

        h = (self.skip_connection(x) + h) / np.sqrt(2.0)
        return self.attn(h)


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    downscale_freq_shift: "float" = 0,
    max_period: int = 10000,
):
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb
