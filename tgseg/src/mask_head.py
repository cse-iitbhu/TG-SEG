import torch
import torch.nn as nn
import torch.nn.functional as F


def build_rel_coords(H, W, device):
    
    y = torch.linspace(-1, 1, steps=H, device=device)
    x = torch.linspace(-1, 1, steps=W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack([xx, yy], dim=0).unsqueeze(0) 
    return coords


class PaperDynamicMaskHead(nn.Module):
   

    def __init__(self, hidden_dim=256, dyn_channels=8, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dyn_channels = dyn_channels
        self.num_layers = num_layers

        in_ch = hidden_dim + 2

        self.num_params = (
            dyn_channels * in_ch + dyn_channels +
            dyn_channels * dyn_channels + dyn_channels +
            1 * dyn_channels + 1
        )

       

        self.controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_params)
        )


        self.level_proj = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
        ])

    def fuse_multiscale(self, feats):
        
        base = feats[0]
        B, D, H, W = base.shape
        out = self.level_proj[0](base)

        for i in range(1, len(feats)):
            fi = self.level_proj[i](feats[i])
            fi = F.interpolate(fi, size=(H, W), mode="bilinear", align_corners=False)
            out = out + fi
        return out  # [B,D,H,W]

    def parse_params(self, params, in_ch):
        
        B, Q, P = params.shape
        dc = self.dyn_channels

        w1_sz = dc * in_ch
        b1_sz = dc
        w2_sz = dc * dc
        b2_sz = dc
        w3_sz = 1 * dc
        b3_sz = 1

        offset = 0

        w1 = params[:, :, offset:offset+w1_sz].reshape(B * Q * dc, in_ch, 1, 1)
        offset += w1_sz
        b1 = params[:, :, offset:offset+b1_sz].reshape(B * Q * dc)
        offset += b1_sz

        w2 = params[:, :, offset:offset+w2_sz].reshape(B * Q * dc, dc, 1, 1)
        offset += w2_sz
        b2 = params[:, :, offset:offset+b2_sz].reshape(B * Q * dc)
        offset += b2_sz

        w3 = params[:, :, offset:offset+w3_sz].reshape(B * Q * 1, dc, 1, 1)
        offset += w3_sz
        b3 = params[:, :, offset:offset+b3_sz].reshape(B * Q * 1)
        offset += b3_sz

        return (w1, b1, w2, b2, w3, b3)


    def forward(self, query_feats, fpn_feats):
        
        B, Q, D = query_feats.shape
        device = query_feats.device

        fused = self.fuse_multiscale(fpn_feats) 
        B, D, H, W = fused.shape

        coords = build_rel_coords(H, W, device).repeat(B, 1, 1, 1)  
        x = torch.cat([fused, coords], dim=1)  
        in_ch = D + 2

        params = self.controller(query_feats)  
        w1, b1, w2, b2, w3, b3 = self.parse_params(params, in_ch)

        x = x.unsqueeze(1).repeat(1, Q, 1, 1, 1)        
        x = x.view(1, B*Q*in_ch, H, W)                  

        
        x = F.conv2d(x, w1, b1, groups=B*Q)              
        x = F.relu(x)

        
        x = F.conv2d(x, w2, b2, groups=B*Q)             
        x = F.relu(x)

        
        x = F.conv2d(x, w3, b3, groups=B*Q)             

        masks = x.view(B, Q, H, W)
        return masks



