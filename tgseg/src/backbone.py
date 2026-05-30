import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)

        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  
        self.layer2 = resnet.layer2  
        self.layer3 = resnet.layer3  
        self.layer4 = resnet.layer4  

        self.out_channels = [256, 512, 1024, 2048]

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]


class FPNProjection(nn.Module):
    
    def __init__(self, in_channels, hidden_dim=256):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Conv2d(c, hidden_dim, kernel_size=1)
            for c in in_channels
        ])

    def forward(self, feats):
        return [p(f) for p, f in zip(self.proj, feats)]

