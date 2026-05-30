import torch

def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return torch.stack([x0, y0, x1, y1], dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    w  = (x1 - x0).clamp(min=0)
    h  = (y1 - y0).clamp(min=0)
    return torch.stack([cx, cy, w, h], dim=-1)

def box_area(boxes):
    return ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) *
            (boxes[:, 3] - boxes[:, 1]).clamp(min=0))

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  
    wh = (rb - lt).clamp(min=0)                         
    inter = wh[:, :, 0] * wh[:, :, 1]                   

    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou, union

def generalized_box_iou(boxes1, boxes2):
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area_c = wh[:, :, 0] * wh[:, :, 1]

    giou = iou - (area_c - union) / (area_c + 1e-6)
    return giou
