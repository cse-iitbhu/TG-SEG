from __future__ import annotations

from typing import List, Sequence, Tuple

import torch.nn as nn

from src.models.blocks import ConvBlock3D, DownsampleBlock3D, UpsampleBlock3D


class NNUNetLikeBackbone3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        channel_multipliers: Sequence[int],
        strides: Sequence[Sequence[int]],
    ):
        super().__init__()
        channels = [base_channels * m for m in channel_multipliers]
        self.channels = channels
        self.stem = ConvBlock3D(in_channels, channels[0])
        self.down_blocks = nn.ModuleList()
        for i, stride in enumerate(strides):
            self.down_blocks.append(DownsampleBlock3D(channels[i], channels[i + 1], stride=stride))
        self.up_blocks = nn.ModuleList()
        for i in range(len(strides) - 1, -1, -1):
            self.up_blocks.append(
                UpsampleBlock3D(
                    in_channels=channels[i + 1],
                    skip_channels=channels[i],
                    out_channels=channels[i],
                    stride=strides[i],
                )
            )

    def forward_encoder(self, x):
        skips = []
        x = self.stem(x)
        skips.append(x)
        for block in self.down_blocks:
            x = block(x)
            skips.append(x)
        bottleneck = skips[-1]
        encoder_skips = skips[:-1]
        return bottleneck, encoder_skips

    def forward_decoder(self, bottleneck, encoder_skips):
        x = bottleneck
        decoder_features = []
        for block, skip in zip(self.up_blocks, reversed(encoder_skips)):
            x = block(x, skip)
            decoder_features.append(x)
        final_feature = x
        return decoder_features, final_feature
