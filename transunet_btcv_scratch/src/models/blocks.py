from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, out_channels, kernel_size=3, stride=1),
            ConvNormAct(out_channels, out_channels, kernel_size=3, stride=1),
        )

    def forward(self, x):
        return self.block(x)


class DownsampleBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: Sequence[int]):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=tuple(stride), stride=tuple(stride), bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.01),
        )
        self.conv = ConvBlock3D(out_channels, out_channels)

    def forward(self, x):
        x = self.down(x)
        return self.conv(x)


class UpsampleBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, stride: Sequence[int]):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=tuple(stride), stride=tuple(stride))
        self.conv = ConvBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return self.gamma * x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0, layerscale: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)
        self.ls1 = LayerScale(dim) if layerscale else nn.Identity()
        self.ls2 = LayerScale(dim) if layerscale else nn.Identity()

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + self.ls1(attn_out)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class DecoderSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x):
        sa_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + sa_out
        x = x + self.mlp(self.norm2(x))
        return x


class SinePositionEmbedding3D(nn.Module):
    def __init__(self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = True, scale: float = 2 * math.pi):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, d, h, w = x.shape
        z_embed = torch.linspace(0, 1, d, device=x.device).view(1, d, 1, 1).expand(b, d, h, w)
        y_embed = torch.linspace(0, 1, h, device=x.device).view(1, 1, h, 1).expand(b, d, h, w)
        x_embed = torch.linspace(0, 1, w, device=x.device).view(1, 1, 1, w).expand(b, d, h, w)
        if self.normalize:
            z_embed = z_embed * self.scale
            y_embed = y_embed * self.scale
            x_embed = x_embed * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        def _pe(embed):
            pos = embed[..., None] / dim_t
            pos = torch.stack((pos[..., 0::2].sin(), pos[..., 1::2].cos()), dim=-1).flatten(-2)
            return pos

        pos_z = _pe(z_embed)
        pos_y = _pe(y_embed)
        pos_x = _pe(x_embed)
        pos = torch.cat([pos_z, pos_y, pos_x], dim=-1)
        pos = pos.permute(0, 4, 1, 2, 3).contiguous()
        return pos
