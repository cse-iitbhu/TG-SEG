from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import TransformerEncoderBlock


class TransformerEncoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        patch_size: Sequence[int] = (1, 1, 1),
        nominal_grid_size: Sequence[int] = (5, 14, 12),
        dropout: float = 0.0,
        use_layerscale: bool = True,
    ):
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.proj_in = nn.Conv3d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *nominal_grid_size))
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    layerscale=use_layerscale,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        if self.patch_size == (1, 1, 1):
            self.proj_out = nn.Conv3d(embed_dim, in_channels, kernel_size=1, stride=1)
        else:
            self.proj_out = nn.ConvTranspose3d(embed_dim, in_channels, kernel_size=self.patch_size, stride=self.patch_size)

    def _get_pos_embed(self, shape: Tuple[int, int, int]) -> torch.Tensor:
        if tuple(self.pos_embed.shape[2:]) == tuple(shape):
            return self.pos_embed
        return F.interpolate(self.pos_embed, size=shape, mode='trilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.proj_in(x)
        b, c, d, h, w = x.shape
        pos = self._get_pos_embed((d, h, w))
        x = x + pos
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(b, c, d, h, w)
        x = self.proj_out(x)
        if x.shape[2:] != residual.shape[2:]:
            x = F.interpolate(x, size=residual.shape[2:], mode='trilinear', align_corners=False)
        return x + residual
