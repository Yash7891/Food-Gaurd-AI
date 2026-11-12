# train_spoilage_multi.py
"""Fine‑tune the spoilage classifier using **multiple** independent datasets.

Usage example::

    python scripts/train_spoilage_multi.py \
        --data-dir C:/path/to/dataset1 \
        --data-dir C:/path/to/dataset2 \
        --epochs 12 \
        --out spoilage_multi.pt

Each supplied directory must have the following layout:

    <root>/fresh/<image files>
    <root>/stale/<image files>

The script concatenates all datasets, applies the same augmentation pipeline we use for
single‑dataset training, and fine‑tunes the existing MobileNetV2 checkpoint.
"""

import argparse
import torch
import torchvision.transforms as T
from torchvision import models
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------
# Dataset definition – reads a single ``root`` with ``fresh``/``stale`` subfolders.
# ---------------------------------------------------------------------------
class SpoilageDataset(Dataset):
    def __init__(self, root: Path, transform):
        self.samples = []
        for label, idx in [("fresh", 0), ("stale", 1)]:
            for img_path in (root / label).rglob("*.[jp][pn]g"):
                self.samples.append((img_path, idx))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, target = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), target

# ---------------------------------------------------------------------------
# Augmentor – same as before with optional synthetic brown spots.
# ---------------------------------------------------------------------------
class SpoilageAugmentor:
    def __call__(self, img: Image.Image) -> Image.Image:
        # colour jitter
        jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05)
        img = jitter(img)
        # occasional synthetic brown spot (adds robustness to tiny patches)
        if torch.rand(1).item() < 0.3:
            import cv2, numpy as np
            arr = np.array(img)
            h, w, _ = arr.shape
            r = int(torch.randint(5, 15, (1,)).item())
            cx = int(torch.randint(r, w - r, (1,)).item())
            cy = int(torch.randint(r, h - r, (1,)).item())
            cv2.circle(arr, (cx, cy), r, (30, 30, 30), -1)  # dark brown spot
            img = Image.fromarray(arr)
        return img

# ---------------------------------------------------------------------------
# Main training routine – builds a concatenated dataset from all ``--data-dir``
# arguments and fine‑tunes the MobilenetV2 checkpoint.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine‑tune spoilage model on multiple datasets")
    parser.add_argument("--data-dir", action="append", required=True,
                        help="Root folder of a dataset (must contain fresh/ and stale/ subfolders). You can specify this flag multiple times.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", type=str, default="spoilage_multi.pt",
                        help="File name for the fine‑tuned checkpoint.")
    args = parser.parse_args()

    # Compose the same transform pipeline used for single‑dataset training.
    tf = T.Compose([
        SpoilageAugmentor(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Build a list of SpoilageDataset objects – one per supplied root.
    datasets = []
    for d in args.data_dir:
        root = Path(d)
        if not (root / "fresh").exists() or not (root / "stale").exists():
            raise RuntimeError(f"Dataset at {root} does not contain required fresh/ and stale/ sub‑folders")
        datasets.append(SpoilageDataset(root, tf))
    # Concatenate them into a single training set.
    train_set = ConcatDataset(datasets)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    # Load the base MobileNetV2 checkpoint (if present). Otherwise start from scratch.
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
    base_ckpt = Path("spoilage_mobilenetv2.pt")
    if base_ckpt.exists():
        model.load_state_dict(torch.load(base_ckpt, map_location="cpu"))
        print(f"Loaded base checkpoint from {base_ckpt}")
    else:
        print("Base checkpoint not found – training from random init")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            _, pred_labels = preds.max(1)
            correct += (pred_labels == yb).sum().item()
            total += xb.size(0)
        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100
        print(f"Epoch {epoch:02d}/{args.epochs} – loss {epoch_loss:.4f} – acc {epoch_acc:.2f}%")

    torch.save(model.state_dict(), args.out)
    print(f"✅ Fine‑tuned model saved to {args.out}")

if __name__ == "__main__":
    main()

