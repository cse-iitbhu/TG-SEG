import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalFusion(nn.Module):
    

    def __init__(self, hidden_dim=256, num_heads=8):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.weight_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, vision_tokens_by_level, text_tokens_by_prompt, text_mask):
        

        num_levels  = len(vision_tokens_by_level)
        num_prompts = len(text_tokens_by_prompt)

        B = vision_tokens_by_level[0].size(0)
        device = vision_tokens_by_level[0].device

        proposals = []   
        weights   = []   

        for l in range(num_levels):
            V_l = vision_tokens_by_level[l]
            level_props = []
            level_wts   = []

            for p in range(num_prompts):
                T_p = text_tokens_by_prompt[p]
                M_p = text_mask[p]

                A_lp, _ = self.attn(
                    query=V_l,
                    key=T_p,
                    value=T_p,
                    key_padding_mask=M_p
                )   

              
                pooled = A_lp.mean(dim=1)    
                w_lp   = self.weight_mlp(pooled)

                level_props.append(A_lp)
                level_wts.append(w_lp)

            proposals.append(level_props)
            weights.append(level_wts)

        fused_tokens_by_level = []
        prompt_weights = []

        for l in range(num_levels):
            props_l = proposals[l]      
            wts_l   = weights[l]        

            wts = torch.cat(wts_l, dim=1)

            wts = F.softmax(wts, dim=1)

            fused = 0.0
            for p in range(num_prompts):
                w = wts[:, p:p+1].unsqueeze(-1)  
                fused = fused + w * props_l[p]

            fused_tokens_by_level.append(fused)
            prompt_weights.append(wts)

        prompt_weights = torch.stack(prompt_weights, dim=0).permute(1, 0, 2)

        return fused_tokens_by_level, prompt_weights


