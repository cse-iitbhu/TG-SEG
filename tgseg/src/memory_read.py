import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskMemoryRead(nn.Module):
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.key_proj   = nn.Conv2d(hidden_dim + 1, hidden_dim, 3, padding=1)
        self.value_proj = nn.Conv2d(hidden_dim + 1, hidden_dim, 3, padding=1)
        self.query_proj = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.scale = hidden_dim ** -0.5

    def forward(self, curr_feat, prev_feat, prev_mask):
        
        B, D, H, W = curr_feat.shape

        memory = torch.cat([prev_feat, prev_mask.unsqueeze(1)], dim=1) 

        Q = self.query_proj(curr_feat)  
        K = self.key_proj(memory)        
        V = self.value_proj(memory)     

        Q = Q.flatten(2).transpose(1, 2)  
        K = K.flatten(2)                  
        V = V.flatten(2).transpose(1, 2)   

        attn = torch.bmm(Q, K) * self.scale   
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, V)            
        out = out.transpose(1, 2).reshape(B, D, H, W)

        return out

