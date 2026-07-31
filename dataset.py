import os
import re
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=15, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class ClothingSegDataset(Dataset):
    """
    Expects:
        images_dir/img_0001.png, img_0002.png, ...
        masks_dir/seg_0001.png,  seg_0002.png,  ...

    Masks are converted to binary: 0 = background, 1 = foreground (clothing).
    Any pixel value > 0 in the mask is treated as foreground. If your masks
    are already strictly 0/1 this is a no-op; if they're 0/255 this handles
    it too.
    """

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.image_files = sorted(
            [f for f in os.listdir(images_dir) if f.lower().endswith(".png")]
        )
        self.mask_files = sorted(
            [f for f in os.listdir(masks_dir) if f.lower().endswith(".png")]
        )

        if len(self.image_files) != len(self.mask_files):
            print(
                f"WARNING: {len(self.image_files)} images vs "
                f"{len(self.mask_files)} masks — check for missing pairs."
            )

        # Build pairs by matching the numeric id: img_0001.png <-> seg_0001.png
        self.pairs = []
        mask_lookup = {self._extract_id(f): f for f in self.mask_files}
        for img_f in self.image_files:
            img_id = self._extract_id(img_f)
            if img_id in mask_lookup:
                self.pairs.append((img_f, mask_lookup[img_id]))
            else:
                print(f"WARNING: no mask found for {img_f}, skipping.")

        if len(self.pairs) == 0:
            raise RuntimeError(
                "No image/mask pairs matched. Check filename patterns "
                "(expected img_XXXX.png / seg_XXXX.png)."
            )

    @staticmethod
    def _extract_id(filename):
        match = re.search(r"(\d+)", filename)
        return match.group(1) if match else filename

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]

        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Binarize: any non-zero pixel = foreground (1)
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = mask.unsqueeze(0) if mask.dim() == 2 else mask  # (1, H, W)

        return image, mask