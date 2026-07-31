import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def iou_score(preds, targets, threshold=0.5, eps=1e-7):
    """
    preds:   raw logits, shape (B, 1, H, W)
    targets: binary masks, shape (B, 1, H, W)
    Returns mean IoU over the batch.
    """
    preds = (torch.sigmoid(preds) > threshold).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def dice_coefficient(preds, targets, threshold=0.5, eps=1e-7):
    preds = (torch.sigmoid(preds) > threshold).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps)
    return dice.mean().item()


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss — works well for binary segmentation."""

    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, preds, targets, eps=1e-7):
        bce_loss = self.bce(preds, targets)

        probs = torch.sigmoid(preds)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        dice = (2 * intersection + eps) / (
            probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps
        )
        dice_loss = 1 - dice.mean()

        return self.weight_bce * bce_loss + self.weight_dice * dice_loss


def save_checkpoint(model, optimizer, epoch, path="checkpoint.pth"):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
    print(f"Checkpoint saved: {path}")


def load_checkpoint(model, optimizer, path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0)


def visualize_prediction(image, mask, pred, threshold=0.5, save_path=None):
    """
    image: tensor (3, H, W), normalized
    mask:  tensor (1, H, W)
    pred:  raw logits tensor (1, H, W)
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    mask_np = mask.cpu().squeeze().numpy()
    pred_np = (torch.sigmoid(pred.cpu()) > threshold).float().squeeze().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Ground Truth")
    axes[2].imshow(pred_np, cmap="gray")
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()