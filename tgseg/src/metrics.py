
import torch

def dice_score(pred, target, eps=1e-6):
    
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    target = target.to(device=pred.device, dtype=pred.dtype)

    pred = pred.sigmoid()
    num = 2 * (pred * target).sum(dim=[1, 2])
    den = pred.sum(dim=[1, 2]) + target.sum(dim=[1, 2]) + eps
    return (num / den).mean().item()


def iou_score(pred, target, eps=1e-6):
    
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if target.dim() == 2:
        target = target.unsqueeze(0)

    target = target.to(device=pred.device, dtype=pred.dtype)

    pred = pred.sigmoid()
    pred = (pred > 0.5).float()

    inter = (pred * target).sum(dim=[1, 2])
    union = pred.sum(dim=[1, 2]) + target.sum(dim=[1, 2]) - inter + eps

    return (inter / union).mean().item()



