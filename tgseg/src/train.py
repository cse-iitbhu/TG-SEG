# Core TG-SEG training script with presence-aware supervision

import torch
from dataset import RefMISS_SpleenDataset
from model import TPPMini
from train_one_epoch import train_one_epoch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = RefMISS_SpleenDataset(
        "../data/custom_dataset/train",
        split="train",
        clip_len=3,
        augment=True,
        max_size=640,
        crop_size=512,
        filter_empty=False,     
        require_all=False
    )

    model = TPPMini(
        hidden_dim=256,
        num_queries=5,
        use_propagation=True,
        lambda_presence=1.0,
        presence_tau=0.5,
        lambda_empty=0.0
    ).to(device)

    
    for p in model.backbone.parameters():
        p.requires_grad = True

    for p in model.text_encoder.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[3],
        gamma=0.1
    )

    num_epochs =5

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} | lr={optimizer.param_groups[0]['lr']:.2e}")

        avg_loss, avg_dice, avg_iou = train_one_epoch(
            model, ds, optimizer, device, batch_size=1, num_workers=0
        )

        print(
            f"Epoch {epoch+1} summary | "
            f"loss={avg_loss:.4f} | dice={avg_dice:.3f} | iou={avg_iou:.3f}"
        )

        scheduler.step(epoch + 1)

    torch.save(model.state_dict(), "tppmini_with_presence_5epoch_lr1e5.pth")
    print("\nModel saved to tppmini_with_presence_5epoch_lr1e5.pth")


if __name__ == "__main__":
    main()
