import torch
import torch.nn.functional as F


def dice_loss(pred, target, eps=1e-6):
   
    pred = pred.sigmoid()

    if pred.dim() == 4:
        num = 2 * (pred * target).sum(dim=[2, 3])
        den = pred.sum(dim=[2, 3]) + target.sum(dim=[2, 3]) + eps
        loss = 1 - (num / den)
        return loss.mean()
    else:
        num = 2 * (pred * target).sum(dim=[1, 2])
        den = pred.sum(dim=[1, 2]) + target.sum(dim=[1, 2]) + eps
        loss = 1 - (num / den)
        return loss.mean()


def bce_loss(pred, target):
    
    return F.binary_cross_entropy_with_logits(pred, target)


def mask_loss(pred, target):
    
    assert pred.dim() == 3, f"mask_loss expects [B,H,W], got {pred.shape}"
    return bce_loss(pred, target) + dice_loss(pred, target)



def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    
    if inputs.dim() == 3:
        inputs = inputs.squeeze(-1)
    if targets.dim() == 3:
        targets = targets.squeeze(-1)

    p = inputs.sigmoid()
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)

    if alpha is not None:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


def mask_loss_per_query(pred_logits_bqhw, target_bhw, eps=1e-6):
    
    assert pred_logits_bqhw.dim() == 4, f"expected [B,Q,H,W], got {pred_logits_bqhw.shape}"
    assert target_bhw.dim() == 3, f"expected [B,H,W], got {target_bhw.shape}"

    B, Q, H, W = pred_logits_bqhw.shape
    target = target_bhw.unsqueeze(1).expand(B, Q, H, W)

    # BCE per query
    bce = F.binary_cross_entropy_with_logits(pred_logits_bqhw, target, reduction="none")
    bce = bce.mean(dim=[2, 3])  # [B,Q]

    # Dice per query
    pred = pred_logits_bqhw.sigmoid()
    inter = (pred * target).sum(dim=[2, 3])  # [B,Q]
    den = pred.sum(dim=[2, 3]) + target.sum(dim=[2, 3]) + eps
    dice = 1 - (2 * inter / den)  # [B,Q]

    return bce + dice

def mask_focal_dice_loss_per_query(pred_logits_bqhw, target_bhw, alpha=0.25, gamma=2.0, eps=1e-6):
    """
    Paper L_mask: Dice + binary mask focal loss (Eq. 13 text).
    pred_logits_bqhw: [B,Q,H,W] logits
    target_bhw:       [B,H,W] {0,1}

    returns:
      per_query_loss: [B,Q]
    """
    assert pred_logits_bqhw.dim() == 4, f"expected [B,Q,H,W], got {pred_logits_bqhw.shape}"
    assert target_bhw.dim() == 3, f"expected [B,H,W], got {target_bhw.shape}"

    B, Q, H, W = pred_logits_bqhw.shape
    target = target_bhw.unsqueeze(1).expand(B, Q, H, W)

    prob = pred_logits_bqhw.sigmoid()
    ce = torch.nn.functional.binary_cross_entropy_with_logits(pred_logits_bqhw, target, reduction="none")
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    focal = ce * ((1.0 - p_t) ** gamma)

    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    focal = alpha_t * focal

    focal = focal.mean(dim=[2, 3]) 

    
    inter = (prob * target).sum(dim=[2, 3])  
    den = prob.sum(dim=[2, 3]) + target.sum(dim=[2, 3]) + eps
    dice = 1.0 - (2.0 * inter / den)        

    return focal + dice



