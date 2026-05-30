import torch

def temporal_iou(masks):
    
    masks = masks.sigmoid()
    masks = (masks > 0.5).float()

    B, T, H, W = masks.shape
    if T < 2:
        return 0.0

    ious = []

    for t in range(1, T):
        m1 = masks[:, t - 1]
        m2 = masks[:, t]

        inter = (m1 * m2).sum(dim=[1, 2])
        union = m1.sum(dim=[1, 2]) + m2.sum(dim=[1, 2]) - inter + 1e-6

        iou = (inter / union).mean().item()
        ious.append(iou)

    return sum(ious) / len(ious)
