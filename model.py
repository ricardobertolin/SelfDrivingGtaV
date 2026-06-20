"""
Two CNN variants for GTA V self-driving.

OffsetCNN  (recommended)
    Input  : 160×90 RGB frame
    Output : single float — predicted lane offset in pixels
             (same semantics as SteeringController.calculate_offset)
    Training: HuberLoss on normalised offset (÷ FRAME_WIDTH/2)
    Inference: predict_offset() de-normalises and returns pixel value

DrivingCNN  (behavioural-cloning fallback)
    Input  : 160×90 RGB frame
    Output : 4 logits [W, A, S, D]
    Training: BCEWithLogitsLoss with class-weight compensation
    Inference: predict() → {W/A/S/D: bool}
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

NUM_ACTIONS = 4    # [W, A, S, D]
INPUT_C     = 3    # RGB
INPUT_H     = 90
INPUT_W     = 160


class DrivingCNN(nn.Module):
    """
    Three Conv-BN-ReLU blocks with stride-2 downsampling, followed by
    AdaptiveAvgPool and two FC layers.  ~1.3 M parameters — fast enough
    for real-time CPU inference at 30 fps.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # 160×90 → 80×45
            nn.Conv2d(INPUT_C, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 80×45 → 40×23
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 40×23 → 20×12
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # spatial pool → 6×6 regardless of minor input-size drift
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_ACTIONS),
            # No sigmoid here — BCEWithLogitsLoss applies it during training.
            # predict() applies sigmoid explicitly at inference time.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    # ------------------------------------------------------------------ #
    #  Inference helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def preprocess(frame_rgb: np.ndarray) -> torch.Tensor:
        """
        Convert a uint8 (INPUT_H × INPUT_W × 3) RGB array to a
        normalised (1 × 3 × INPUT_H × INPUT_W) float32 tensor.
        The frame must already be resized to (INPUT_W, INPUT_H).
        """
        t = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)
        return t.permute(2, 0, 1).unsqueeze(0)   # HWC → 1CHW

    @torch.no_grad()
    def predict(self, frame_rgb: np.ndarray,
                threshold: float = 0.5) -> dict:
        """
        Run a single-frame inference.

        frame_rgb — uint8 array already resized to (INPUT_H, INPUT_W, 3).
        Returns   {'W': bool, 'A': bool, 'S': bool, 'D': bool}.
        """
        logits = self.forward(self.preprocess(frame_rgb)).squeeze(0).cpu()
        probs  = torch.sigmoid(logits).numpy()
        return {k: bool(probs[i] >= threshold)
                for i, k in enumerate(('W', 'A', 'S', 'D'))}

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> Optional['DrivingCNN']:
        path = Path(path)
        if not path.exists():
            print(f"[DrivingCNN] No checkpoint at {path}")
            return None
        model = cls()
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        print(f"[DrivingCNN] Loaded from {path}")
        return model


# Shared normalisation constant: raw pixel offset ÷ OFFSET_NORM ≈ [-0.5, +0.5]
OFFSET_NORM = 400.0   # = FRAME_WIDTH / 2


class OffsetCNN(nn.Module):
    """
    Regression CNN: raw frame → predicted lane centre offset (pixels).

    The output has the same sign convention as
    SteeringController.calculate_offset():
        positive → lane centre is right of image centre → steer right
        negative → lane centre is left  of image centre → steer left

    The model outputs a single normalised float (÷ OFFSET_NORM).
    predict_offset() de-normalises back to pixel units so it can be
    fed directly into SteeringController.decide_action() / decide_speed()
    without any extra wiring.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(INPUT_C, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((6, 6)),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),   # single normalised offset — no activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(x)).squeeze(-1)   # (B,)

    @staticmethod
    def preprocess(frame_rgb: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(frame_rgb.astype(np.float32) / 255.0)
        return t.permute(2, 0, 1).unsqueeze(0)

    @torch.no_grad()
    def predict_offset(self, frame_rgb: np.ndarray) -> float:
        """
        Run inference on one uint8 RGB frame (already resized to INPUT_W×INPUT_H).
        Returns the predicted lane offset in pixels (de-normalised).
        """
        return float(self.forward(self.preprocess(frame_rgb)).item()) * OFFSET_NORM

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> Optional['OffsetCNN']:
        path = Path(path)
        if not path.exists():
            print(f"[OffsetCNN] No checkpoint at {path}")
            return None
        model = cls()
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        print(f"[OffsetCNN] Loaded from {path}")
        return model
