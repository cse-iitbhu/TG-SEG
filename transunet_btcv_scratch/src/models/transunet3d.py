from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from src.models.nnunet3d_backbone import NNUNetLikeBackbone3D
from src.models.transformer_decoder import QueryMaskDecoder3D
from src.models.transformer_encoder import TransformerEncoder3D


class TransUNet3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mode = cfg.model.mode
        self.num_classes = int(cfg.model.num_classes)

        self.backbone = NNUNetLikeBackbone3D(
            in_channels=int(cfg.model.in_channels),
            base_channels=int(cfg.model.base_channels),
            channel_multipliers=list(cfg.model.channel_multipliers),
            strides=list(cfg.model.strides),
        )

        channels = self.backbone.channels
        bottleneck_channels = channels[-1]

        self.encoder_transformer = None
        if self.mode in ['encoder_only', 'encoder_decoder']:
            nominal_grid = tuple(cfg.model.encoder.nominal_grid)
            self.encoder_transformer = TransformerEncoder3D(
                in_channels=bottleneck_channels,
                embed_dim=int(cfg.model.encoder.embed_dim),
                depth=int(cfg.model.encoder.depth),
                num_heads=int(cfg.model.encoder.num_heads),
                mlp_ratio=float(cfg.model.encoder.mlp_ratio),
                patch_size=tuple(cfg.model.encoder.patch_size),
                nominal_grid_size=nominal_grid,
                dropout=float(cfg.model.encoder.dropout),
                use_layerscale=bool(cfg.model.encoder.use_layerscale),
            )

        self.seg_head = None
        self.decoder_transformer = None
        if self.mode == 'encoder_only':
            self.seg_head = nn.Conv3d(channels[0], self.num_classes, kernel_size=1)
        else:
            decoder_feature_channels = list(reversed(channels[:-1]))
            self.decoder_transformer = QueryMaskDecoder3D(
                in_feature_channels=decoder_feature_channels,
                num_classes=self.num_classes - 1,
                num_queries=int(cfg.model.decoder.num_queries),
                hidden_dim=int(cfg.model.decoder.hidden_dim),
                num_layers=int(cfg.model.decoder.num_layers),
                num_heads=int(cfg.model.decoder.num_heads),
                dropout=float(cfg.model.decoder.dropout),
                use_cross_attention=bool(cfg.model.decoder.use_cross_attention),
                use_multiscale_features=bool(cfg.model.decoder.use_multiscale_features),
                use_masked_attention=bool(cfg.model.decoder.use_masked_attention),
                use_query_positional_embeddings=bool(cfg.model.decoder.use_query_positional_embeddings),
            )

    def forward(self, x: torch.Tensor) -> dict:
        bottleneck, skips = self.backbone.forward_encoder(x)
        if self.encoder_transformer is not None:
            bottleneck = self.encoder_transformer(bottleneck)
        decoder_features, final_feature = self.backbone.forward_decoder(bottleneck, skips)

        if self.mode == 'encoder_only':
            logits = self.seg_head(final_feature)
            return {'semantic_logits': logits}

        out = self.decoder_transformer(decoder_features, final_feature)
        return out
