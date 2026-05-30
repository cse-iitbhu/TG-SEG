
import argparse
import torch
import torch.nn.functional as F

from dataset import RefMISS_SpleenDataset
from model import TPPMini
from metrics import dice_score, iou_score
from temporal_metrics import temporal_iou


def dice_iou_presence_aware(pred_logits, gt_bin, pred_present: bool):
    
    gt_empty = (gt_bin.sum().item() == 0.0)

    if (not pred_present) and gt_empty:
        return 1.0, 1.0
    if (not pred_present) and (not gt_empty):
        return 0.0, 0.0
    if pred_present and gt_empty:
        return 0.0, 0.0

    d = float(dice_score(pred_logits, gt_bin))
    j = float(iou_score(pred_logits, gt_bin))
    return d, j


@torch.no_grad()
def evaluate_dataset(
    ckpt_path: str,
    data_root: str,
    use_propagation: bool = True,
    clip_limit: int | None = None,
    filter_empty: bool = False,
    require_all: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = RefMISS_SpleenDataset(
        data_root,
        split="train",
        augment=False,
        filter_empty=filter_empty,   
        require_all=require_all,
    )

    model = TPPMini(
        hidden_dim=256,
        num_queries=5,
        use_propagation=use_propagation,
        lambda_presence=1.0,
        presence_tau=0.5,
        lambda_empty=0.0
    ).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    total_dice_all = 0.0
    total_iou_all = 0.0
    total_tiou = 0.0

    total_dice_fg = 0.0
    total_iou_fg = 0.0
    used_fg = 0

    tp = fp = tn = fn = 0

    total_frames = 0
    total_clips = 0

    n = len(ds) if clip_limit is None else min(len(ds), clip_limit)

    for i in range(n):
        imgs, target = ds[i]
        imgs = imgs.unsqueeze(0).to(device)                 
        gt_masks = target["masks"].unsqueeze(0).to(device)  

        captions = [target["caption"]]  

        masks, _, scores, presence = model.forward_sequence(imgs, captions)
        T = masks.shape[1]

        best_masks = []
        for t in range(T):
            best_q = int(scores[0, t].argmax().item())
            best_masks.append(masks[0, t, best_q]) 
        best_masks = torch.stack(best_masks, dim=0).unsqueeze(0) 

        for t in range(T):
            pred = best_masks[:, t]  
            gt_ds = F.interpolate(
                gt_masks[:, t].unsqueeze(1),
                size=pred.shape[-2:],
                mode="nearest"
            ).squeeze(1)  

            pred_present = (torch.sigmoid(presence[0, t]).item() >= model.presence_tau)
            gt_present = (gt_ds.sum().item() > 0.0)

            # presence confusion
            if pred_present and gt_present:
                tp += 1
            elif pred_present and (not gt_present):
                fp += 1
            elif (not pred_present) and (not gt_present):
                tn += 1
            else:
                fn += 1

            d_all, j_all = dice_iou_presence_aware(pred, gt_ds, pred_present)
            total_dice_all += d_all
            total_iou_all += j_all
            total_frames += 1

            if gt_present:
                total_dice_fg += float(dice_score(pred, gt_ds))
                total_iou_fg  += float(iou_score(pred, gt_ds))
                used_fg += 1

        total_tiou += float(temporal_iou(best_masks))
        total_clips += 1

    mean_dice_all = total_dice_all / max(total_frames, 1)
    mean_iou_all  = total_iou_all  / max(total_frames, 1)
    mean_tiou     = total_tiou     / max(total_clips, 1)

    mean_dice_fg  = total_dice_fg / max(used_fg, 1)
    mean_iou_fg   = total_iou_fg  / max(used_fg, 1)

    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = (2 * prec * rec) / max(prec + rec, 1e-12)
    acc  = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "dice_all": mean_dice_all,
        "iou_all": mean_iou_all,
        "tiou": mean_tiou,
        "dice_fg": mean_dice_fg,
        "iou_fg": mean_iou_fg,
        "presence_acc": acc,
        "presence_prec": prec,
        "presence_rec": rec,
        "presence_f1": f1,
        "frames": total_frames,
        "fg_frames": used_fg,
        "clips": total_clips,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="../data/custom_dataset/train")
    parser.add_argument("--no_prop", action="store_true", help="Disable propagation at eval")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only first N clips (0 = full)")
    parser.add_argument("--filter_empty", action="store_true", help="Filter empty-object clips (NOT recommended for realistic eval)")
    parser.add_argument("--require_all", action="store_true", help="If filtering, require object in ALL frames")
    args = parser.parse_args()

    use_propagation = (not args.no_prop)
    clip_limit = None if args.limit <= 0 else args.limit

    r = evaluate_dataset(
        ckpt_path=args.ckpt,
        data_root=args.data_root,
        use_propagation=use_propagation,
        clip_limit=clip_limit,
        filter_empty=args.filter_empty,
        require_all=args.require_all,
    )

    print("\n=== Evaluation (Presence-gated, presence-aware empty Dice) ===")
    print(f"ckpt        : {args.ckpt}")
    print(f"data_root   : {args.data_root}")
    print(f"propagation : {use_propagation}")
    print(f"clips       : {r['clips']} | frames={r['frames']} | fg_frames={r['fg_frames']}")
    print(f"presence CM : TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']}")

    print("\nAll-frame (presence-aware empty):")
    print(f"  Dice_all : {r['dice_all']:.3f}")
    print(f"  IoU_all  : {r['iou_all']:.3f}")
    print(f"  Temp-IoU : {r['tiou']:.3f}")

    print("\nForeground-only (GT non-empty):")
    print(f"  Dice_fg : {r['dice_fg']:.3f}")
    print(f"  IoU_fg  : {r['iou_fg']:.3f}")

    print("\nPresence (present=positive):")
    print(f"  Acc  : {r['presence_acc']:.3f}")
    print(f"  Prec : {r['presence_prec']:.3f}")
    print(f"  Rec  : {r['presence_rec']:.3f}")
    print(f"  F1   : {r['presence_f1']:.3f}")


if __name__ == "__main__":
    main()



