import torch
import torch.nn as nn
import torch.nn.functional as F

from ms_deform_attn import MSDeformAttn


def build_spatial_shapes_and_start_index(feat_shapes, device):
    spatial_shapes = torch.as_tensor(feat_shapes, dtype=torch.long, device=device)  # [L,2]
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    return spatial_shapes, level_start_index


def build_encoder_reference_points(feat_shapes, device):
    ref_list = []
    L = len(feat_shapes)
    for (H, W) in feat_shapes:
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        )
        ref = torch.stack([(x + 0.5) / W, (y + 0.5) / H], dim=-1) 
        ref_list.append(ref.reshape(-1, 2))  
    ref = torch.cat(ref_list, dim=0) 
    ref = ref[None, :, None, :].repeat(1, 1, L, 1)  
    return ref


class DeformableEncoderLayer(nn.Module):
    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4, dim_ffn=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ffn, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, reference_points, spatial_shapes, level_start_index):
        src2 = self.self_attn(src, reference_points, src, spatial_shapes, level_start_index)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class DeformableEncoder(nn.Module):
    def __init__(self, num_layers=4, d_model=256, n_levels=3, n_heads=8, n_points=4, dim_ffn=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DeformableEncoderLayer(d_model, n_levels, n_heads, n_points, dim_ffn, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src, reference_points, spatial_shapes, level_start_index):
        out = src
        for layer in self.layers:
            out = layer(out, reference_points, spatial_shapes, level_start_index)
        return out


class DeformableDecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4, dim_ffn=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, dim_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ffn, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, reference_points, memory, spatial_shapes, level_start_index):
        q = k = tgt
        tgt2, _ = self.self_attn(q, k, value=tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn(tgt, reference_points, memory, spatial_shapes, level_start_index)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class DeformableDecoder(nn.Module):
    def __init__(self, num_layers=4, d_model=256, n_levels=3, n_heads=8, n_points=4, dim_ffn=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DeformableDecoderLayer(d_model, n_levels, n_heads, n_points, dim_ffn, dropout)
            for _ in range(num_layers)
        ])
        self.ref_point_head = nn.Linear(d_model, 2)

    def forward(self, tgt, memory, spatial_shapes, level_start_index):
        B, Q, D = tgt.shape
        L = spatial_shapes.shape[0]

        ref = torch.sigmoid(self.ref_point_head(tgt))  
        reference_points = ref[:, :, None, :].repeat(1, 1, L, 1)  

        out = tgt
        for layer in self.layers:
            out = layer(out, reference_points, memory, spatial_shapes, level_start_index)
        return out


class DeformableTransformer(nn.Module):
   
    def __init__(self, d_model=256, num_queries=5, n_levels=3, n_heads=8, n_points=4,
                 num_encoder_layers=4, num_decoder_layers=4, dim_ffn=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.n_levels = n_levels

        self.encoder = DeformableEncoder(num_encoder_layers, d_model, n_levels, n_heads, n_points, dim_ffn, dropout)
        self.decoder = DeformableDecoder(num_decoder_layers, d_model, n_levels, n_heads, n_points, dim_ffn, dropout)

        self.query_propagator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def encode(self, fused_tokens_by_level, feat_shapes):
        
        device = fused_tokens_by_level[0].device
        B = fused_tokens_by_level[0].shape[0]

        src = torch.cat(fused_tokens_by_level, dim=1) 
        spatial_shapes, level_start_index = build_spatial_shapes_and_start_index(feat_shapes, device)
        reference_points_enc = build_encoder_reference_points(feat_shapes, device).repeat(B, 1, 1, 1)

        memory = self.encoder(src, reference_points_enc, spatial_shapes, level_start_index)
        return memory, spatial_shapes, level_start_index

    def decode(self, memory, spatial_shapes, level_start_index, prev_query=None, text_query=None, use_propagation=True):
        
        device = memory.device
        B = memory.shape[0]
        L = spatial_shapes.shape[0]

        if (prev_query is not None) and use_propagation:
            tgt = prev_query.unsqueeze(1)  
        else:
            if text_query is not None:
                tgt = text_query.unsqueeze(1).repeat(1, self.num_queries, 1)  
            else:
                tgt = torch.zeros(B, self.num_queries, self.d_model, device=device)

        hs = self.decoder(tgt, memory, spatial_shapes, level_start_index)
        return hs

    def forward(self, fused_tokens_by_level, feat_shapes, prev_query=None, text_query=None, use_propagation=True):
        memory, spatial_shapes, level_start_index = self.encode(fused_tokens_by_level, feat_shapes)
        hs = self.decode(memory, spatial_shapes, level_start_index, prev_query, text_query, use_propagation)
        return hs

