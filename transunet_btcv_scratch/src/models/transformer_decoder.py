from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import DecoderSelfAttentionBlock


class CrossAttentionUpdate(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if dim % num_heads != 0:
            raise ValueError(f'dim={dim} must be divisible by num_heads={num_heads}')
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_mem = nn.LayerNorm(dim)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        x = x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        return x

    def forward(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self._reshape_heads(self.q_proj(self.norm_q(queries)))
        k = self._reshape_heads(self.k_proj(self.norm_mem(memory)))
        v = self._reshape_heads(self.v_proj(self.norm_mem(memory)))
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_bias is not None:
            attn = attn + attention_bias.unsqueeze(1)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(queries.shape[0], queries.shape[1], -1)
        out = self.out_proj(out)
        return queries + out


class TransformerDecoderLayer3D(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.cross = CrossAttentionUpdate(dim=dim, num_heads=num_heads, dropout=dropout)
        self.self_block = DecoderSelfAttentionBlock(dim=dim, num_heads=num_heads, dropout=dropout)

    def forward(self, queries, memory, attention_bias=None):
        if self.use_cross_attention:
            queries = self.cross(queries, memory, attention_bias=attention_bias)
        queries = self.self_block(queries)
        return queries


class QueryMaskDecoder3D(nn.Module):
    def __init__(
        self,
        in_feature_channels: Sequence[int],
        num_classes: int,
        num_queries: int = 20,
        hidden_dim: int = 192,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_cross_attention: bool = True,
        use_multiscale_features: bool = True,
        use_masked_attention: bool = True,
        use_query_positional_embeddings: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention
        self.use_multiscale_features = use_multiscale_features
        self.use_masked_attention = use_masked_attention
        self.use_query_positional_embeddings = use_query_positional_embeddings

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_embed = nn.Embedding(num_queries, hidden_dim)
        self.feature_projs = nn.ModuleList([nn.Conv3d(c, hidden_dim, kernel_size=1) for c in in_feature_channels])
        self.single_scale_proj = nn.Conv3d(in_feature_channels[-1], hidden_dim, kernel_size=1)
        self.mask_feature_proj = nn.Conv3d(in_feature_channels[-1], hidden_dim, kernel_size=1)
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer3D(
                    dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_cross_attention=use_cross_attention,
                )
                for _ in range(num_layers)
            ]
        )
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)

    def _flatten_memory(self, memory: torch.Tensor) -> torch.Tensor:
        return memory.flatten(2).transpose(1, 2)

    def _compute_masks(self, queries: torch.Tensor, mask_feature: torch.Tensor) -> torch.Tensor:
        b, q, d = queries.shape
        masks = torch.einsum('bqd,bdhwz->bqhwz', queries, mask_feature)
        return masks

    def _build_attention_bias(self, prev_masks: torch.Tensor, target_size: Tuple[int, int, int]) -> torch.Tensor:
        prev_prob = torch.sigmoid(prev_masks)
        prev_prob = F.interpolate(prev_prob, size=target_size, mode='trilinear', align_corners=False)
        prev_binary = (prev_prob > 0.5).float()
        bias = torch.where(prev_binary > 0.5, torch.zeros_like(prev_binary), torch.full_like(prev_binary, -1e4))
        bias = bias.flatten(2)
        return bias

    def semantic_inference(self, mask_logits: torch.Tensor, class_logits: torch.Tensor) -> torch.Tensor:
        mask_prob = torch.sigmoid(mask_logits)
        class_prob = F.softmax(class_logits, dim=-1)
        semantic = torch.einsum('bqdhw,bqk->bkdhw', mask_prob, class_prob)
        return semantic

    def forward(self, decoder_features: List[torch.Tensor], final_feature: torch.Tensor) -> dict:
        if self.use_multiscale_features:
            candidate_features = decoder_features[: self.num_layers]
            candidate_features = [proj(feat) for proj, feat in zip(self.feature_projs[: len(candidate_features)], candidate_features)]
        else:
            candidate_features = [self.single_scale_proj(final_feature) for _ in range(self.num_layers)]
        mask_feature = self.mask_feature_proj(final_feature)

        b = final_feature.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        if self.use_query_positional_embeddings:
            queries = queries + self.query_pos_embed.weight.unsqueeze(0)

        aux_outputs = []
        prev_masks = self._compute_masks(queries, mask_feature)
        for i, layer in enumerate(self.layers):
            feat = candidate_features[i if self.use_multiscale_features else 0]
            memory = self._flatten_memory(feat)
            attention_bias = None
            if self.use_cross_attention and self.use_masked_attention:
                attention_bias = self._build_attention_bias(prev_masks, feat.shape[2:])
            queries = layer(queries, memory, attention_bias=attention_bias)
            mask_logits = self._compute_masks(queries, mask_feature)
            class_logits = self.class_head(queries)
            aux_outputs.append(
                {
                    'pred_masks': mask_logits,
                    'pred_logits': class_logits,
                }
            )
            prev_masks = mask_logits

        final_masks = aux_outputs[-1]['pred_masks'] if aux_outputs else prev_masks
        final_logits = aux_outputs[-1]['pred_logits'] if aux_outputs else self.class_head(queries)
        semantic_logits = self.semantic_inference(final_masks, final_logits)
        return {
            'pred_masks': final_masks,
            'pred_logits': final_logits,
            'aux_outputs': aux_outputs,
            'semantic_logits': semantic_logits,
            'queries': queries,
        }
