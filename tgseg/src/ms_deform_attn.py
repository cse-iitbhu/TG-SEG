import torch
import torch.nn as nn
import torch.nn.functional as F


class MSDeformAttn(nn.Module):
    

    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.d_per_head = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        nn.init.constant_(self.sampling_offsets.bias, 0.)
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)

    def forward(self, query, reference_points, input_flatten, spatial_shapes, level_start_index):
        
        B, Len_q, D = query.shape
        _, Len_in, _ = input_flatten.shape

        value = self.value_proj(input_flatten)  # [B, Len_in, D]
        value = value.view(B, Len_in, self.n_heads, self.d_per_head)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            B, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )

        attn_weights = self.attention_weights(query)
        attn_weights = attn_weights.view(B, Len_q, self.n_heads, self.n_levels * self.n_points)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.view(B, Len_q, self.n_heads, self.n_levels, self.n_points)

        # normalize offsets by spatial shapes
        spatial_shapes_t = spatial_shapes.to(query.device)
        offset_norm = spatial_shapes_t[None, None, None, :, None, :]  # [1,1,1,L,1,2]
        sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_norm

        # output accumulator
        output = torch.zeros(B, Len_q, self.n_heads, self.d_per_head, device=query.device)

        for lvl in range(self.n_levels):
            H_l, W_l = spatial_shapes_t[lvl]
            start = level_start_index[lvl]
            end = start + H_l * W_l

            value_l = value[:, start:end]  
            value_l = value_l.permute(0, 2, 3, 1).contiguous() 
            value_l = value_l.view(B * self.n_heads, self.d_per_head, H_l, W_l)

            grid = sampling_locations[:, :, :, lvl, :, :] 
            grid = grid.permute(0, 2, 1, 3, 4).contiguous() 
            grid = grid.view(B * self.n_heads, Len_q, self.n_points, 2)

            grid = grid * 2 - 1

            sampled = F.grid_sample(
                value_l,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False
            ) 

            sampled = sampled.view(B, self.n_heads, self.d_per_head, Len_q, self.n_points)
            sampled = sampled.permute(0, 3, 1, 4, 2).contiguous()  

            w = attn_weights[:, :, :, lvl, :]  
            w = w.unsqueeze(-1)                

            output = output + (sampled * w).sum(dim=3)  

        output = output.view(B, Len_q, D)
        return self.output_proj(output)
