import os
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import ClothingSegDataset, get_train_transform, get_val_transform
from model import UNet
from utils import DiceBCELoss, iou_score, dice_coefficient, save_checkpoint, visualize_prediction

# ---------------- Config ----------------
IMAGES_DIR = "dataset/images"
MASKS_DIR = "dataset/masks"
IMG_SIZE = 126
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "unet_clothing_seg.pth"
NUM_WORKERS = 2
# -----------------------------------------


def get_dataloaders():
    full_dataset = ClothingSegDataset(IMAGES_DIR, MASKS_DIR, transform=None)
    n_val = int(len(full_dataset) * VAL_SPLIT)
    n_train = len(full_dataset) - n_val

    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    # Apply transforms per-subset via wrapper datasets
    train_ds = ClothingSegDataset(IMAGES_DIR, MASKS_DIR, transform=get_train_transform(IMG_SIZE))
    val_ds = ClothingSegDataset(IMAGES_DIR, MASKS_DIR, transform=get_val_transform(IMG_SIZE))

    train_ds.pairs = [full_dataset.pairs[i] for i in train_subset.indices]
    val_ds.pairs = [full_dataset.pairs[i] for i in val_subset.indices]

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    running_iou = 0.0

    loop = tqdm(loader, desc="Training", leave=False)
    for images, masks in loop:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = loss_fn(preds, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_iou += iou_score(preds, masks)
        loop.set_postfix(loss=loss.item())

    return running_loss / len(loader), running_iou / len(loader)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0

    for images, masks in tqdm(loader, desc="Validating", leave=False):
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = loss_fn(preds, masks)

        running_loss += loss.item()
        running_iou += iou_score(preds, masks)
        running_dice += dice_coefficient(preds, masks)

    n = len(loader)
    return running_loss / n, running_iou / n, running_dice / n


def main():
    print(f"Using device: {DEVICE}")
    train_loader, val_loader = get_dataloaders()

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = DiceBCELoss()

    best_iou = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss, val_iou, val_dice = validate(model, val_loader, loss_fn, DEVICE)

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(model, optimizer, epoch, CHECKPOINT_PATH)
            print(f"  -> New best model saved (IoU: {best_iou:.4f})")

    print(f"\nTraining complete. Best Val IoU: {best_iou:.4f}")

    # Save a few sample predictions for visual sanity check
    os.makedirs("sample_predictions", exist_ok=True)
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(val_loader))
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        preds = model(images)
        for i in range(min(4, images.shape[0])):
            visualize_prediction(
                images[i], masks[i], preds[i],
                save_path=f"sample_predictions/pred_{i}.png"
            )
    print("Sample predictions saved to sample_predictions/")


if __name__ == "__main__":
    main()