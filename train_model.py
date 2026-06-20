"""
Offline training script — handles both dataset types automatically.

Usage
-----
    python train_model.py

Reads every *.npz file from training_data/ and inspects the label shape:
  labels (N, 1)  →  offset regression   →  trains OffsetCNN  →  offset_model.pth
  labels (N, 4)  →  key classification  →  trains DrivingCNN →  model.pth

You can mix sessions freely: if ALL files have (N,1) labels the offset model
is trained; if ALL have (N,4) the key model is trained; mixed datasets raise
an error so you know to separate them.

Offset regression (recommended)
--------------------------------
Records frames paired with the lane detector's computed offset.  The model
learns to reproduce that number from raw pixels, then the existing
SteeringController thresholds handle the rest.  HuberLoss is used instead of
MSE to be robust to occasional bad detections in the training set.

Key classification (behavioural cloning)
-----------------------------------------
Records frames paired with which keys the human pressed.  BCEWithLogitsLoss
with per-key pos_weight counters class imbalance (W is pressed ~80 % of
frames, so a naive model would always predict W=True).
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from model import DrivingCNN, OffsetCNN, OFFSET_NORM, INPUT_H, INPUT_W, NUM_ACTIONS

# ── Global config ──────────────────────────────────────────────────────────────
DATA_DIR         = Path('training_data')
OFFSET_MODEL     = Path('offset_model.pth')
KEYS_MODEL       = Path('model.pth')
MIN_CHUNK_FRAMES = 50    # chunks smaller than this are silently discarded
BATCH_SIZE       = 64
EPOCHS          = 25
LR              = 3e-4
VAL_SPLIT       = 0.15
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Dataset ───────────────────────────────────────────────────────────────────

class DrivingDataset(Dataset):
    """Loads all .npz chunks; auto-detects offset vs keys mode from label shape."""

    def __init__(self, data_dir: Path,
                 min_frames: int = MIN_CHUNK_FRAMES) -> None:
        all_files = sorted(data_dir.glob('*.npz'))
        if not all_files:
            sys.exit(f"No .npz files in {data_dir}. Record training data first (R key).")

        frames_list, labels_list = [], []
        label_widths: set = set()
        skipped = []
        for f in all_files:
            with np.load(f) as d:
                n = d['frames'].shape[0]
                if n < min_frames:
                    skipped.append((f.name, n))
                    continue
                label_widths.add(d['labels'].shape[1])
                frames_list.append(d['frames'].copy())
                labels_list.append(d['labels'].copy())

        if skipped:
            print(f"Skipped {len(skipped)} chunk(s) with < {min_frames} frames:")
            for name, n in skipped:
                print(f"  {name}  ({n} frames)")

        if not frames_list:
            sys.exit(f"All chunks have fewer than {min_frames} frames. "
                     "Record more data or lower MIN_CHUNK_FRAMES.")

        if len(label_widths) > 1:
            sys.exit(
                f"Mixed dataset: found label widths {label_widths} across files.\n"
                "Old key-label chunks (width=4) and new offset-label chunks (width=1) "
                "cannot be concatenated.\n"
                "Move the old *.npz files to a separate folder and re-run."
            )

        self.frames = np.concatenate(frames_list, axis=0)
        self.labels = np.concatenate(labels_list, axis=0)

        used = len(all_files) - len(skipped)
        self.mode = 'offset' if self.labels.shape[1] == 1 else 'keys'
        print(f"Mode: {self.mode}  |  {len(self.frames):,} frames  |  "
              f"{used}/{len(all_files)} file(s) used")

        if self.mode == 'offset':
            print(f"  Offset  mean={self.labels[:,0].mean():.1f}  "
                  f"std={self.labels[:,0].std():.1f}  "
                  f"min={self.labels[:,0].min():.0f}  "
                  f"max={self.labels[:,0].max():.0f}")
        else:
            for i, k in enumerate(('W', 'A', 'S', 'D')):
                pct = self.labels[:, i].mean() * 100
                print(f"  {k}: {pct:.1f} % positive")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(
            self.frames[idx].astype(np.float32) / 255.0
        ).permute(2, 0, 1)                          # HWC → CHW

        if self.mode == 'offset':
            # Normalise offset to ≈ [-0.5, +0.5] for training stability
            y = torch.tensor(self.labels[idx, 0] / OFFSET_NORM, dtype=torch.float32)
        else:
            y = torch.from_numpy(self.labels[idx])  # (4,) float32

        return x, y


# ── Training ──────────────────────────────────────────────────────────────────

def train() -> None:
    print(f"Device: {DEVICE}")
    print(f"Loading data from {DATA_DIR} …\n")

    full_ds = DrivingDataset(DATA_DIR)
    n_val   = max(1, int(len(full_ds) * VAL_SPLIT))
    n_train = len(full_ds) - n_val

    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, num_workers=0)

    if full_ds.mode == 'offset':
        model      = OffsetCNN().to(DEVICE)
        criterion  = nn.HuberLoss(delta=0.2)   # robust to rare bad detections
        save_path  = OFFSET_MODEL
        _train_epoch = _epoch_offset
        _val_epoch   = _val_offset
    else:
        model     = DrivingCNN().to(DEVICE)
        labels_all = full_ds.labels
        n          = len(labels_all)
        pos_count  = labels_all.sum(axis=0).clip(min=1)
        pos_weight = torch.tensor((n - pos_count) / pos_count,
                                  dtype=torch.float32, device=DEVICE)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        save_path  = KEYS_MODEL
        _train_epoch = _epoch_keys
        _val_epoch   = _val_keys
        print(f"\npos_weight: {dict(zip('WASD', pos_weight.cpu().tolist()))}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    best_val = float('inf')
    print()

    for epoch in range(1, EPOCHS + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion)
        val_loss, extra = _val_epoch(model, val_loader, criterion, n_val)

        marker = '  *' if val_loss < best_val else ''
        print(f"Ep {epoch:2d}/{EPOCHS}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"{extra}{marker}")

        if val_loss < best_val:
            best_val = val_loss
            model.save(save_path)

        scheduler.step()

    print(f"\nBest val loss: {best_val:.4f}  ->  {save_path}")


# ── Per-epoch helpers ─────────────────────────────────────────────────────────

def _epoch_offset(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x)
    return total / len(loader.dataset)


@torch.no_grad()
def _val_offset(model, loader, criterion, n_val):
    model.eval()
    total = 0.0
    abs_err = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        total   += criterion(pred, y).item() * len(x)
        abs_err += (pred - y).abs().sum().item()
    loss = total / n_val
    mae_px = abs_err / n_val * OFFSET_NORM   # back to pixels
    return loss, f"MAE={mae_px:.1f}px"


def _epoch_keys(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x)
    return total / len(loader.dataset)


@torch.no_grad()
def _val_keys(model, loader, criterion, n_val):
    model.eval()
    total   = 0.0
    correct = np.zeros(NUM_ACTIONS)
    for x, y in loader:
        x, y  = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        total    += criterion(logits, y).item() * len(x)
        preds     = (torch.sigmoid(logits) >= 0.5).float()
        correct  += (preds == y).float().cpu().numpy().sum(axis=0)
    loss = total / n_val
    acc  = correct / n_val * 100
    extra = f"acc W={acc[0]:.0f}% A={acc[1]:.0f}% S={acc[2]:.0f}% D={acc[3]:.0f}%"
    return loss, extra


if __name__ == '__main__':
    train()
