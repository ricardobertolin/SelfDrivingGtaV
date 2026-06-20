"""
Training data collector for GTA V behavioural cloning.

Records (frame, key_state) pairs to disk during manual play.
Frames are resized to (INPUT_W × INPUT_H) to match the CNN input.
Labels are float32 arrays [W, A, S, D] with 1.0 = key pressed.

Data is written in chunks of CHUNK_SIZE frames to avoid holding
the whole session in memory.  Each chunk is a compressed .npz:
    frames : (N, INPUT_H, INPUT_W, 3)  uint8
    labels : (N, 4)                    float32

Usage (from main.py)
---------------------
    collector = DataCollector()
    collector.start()
    while recording:
        collector.record(frame, {'W': True, 'A': False, 'S': False, 'D': False})
    collector.stop()

Then train offline:
    python train_model.py
"""

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from model import INPUT_H, INPUT_W

CHUNK_SIZE   = 500          # flush every N frames
DATA_DIR     = 'training_data'
OBSTACLE_DIR = 'training_data/obstacles'   # confirmed-obstacle frames (label=1)


class DataCollector:
    """Records screen frames + simultaneous WASD key states to disk."""

    def __init__(self, data_dir: str = DATA_DIR) -> None:
        self._dir    = Path(data_dir)
        self._dir.mkdir(exist_ok=True)
        self._obs_dir = Path(OBSTACLE_DIR)
        self._obs_dir.mkdir(parents=True, exist_ok=True)
        self._frames: list = []
        self._labels: list = []
        self._obs_frames: list = []
        self._obs_chunk   = 0
        self._obs_total   = 0
        self._session = ''
        self._chunk   = 0
        self._total   = 0
        self.active   = False

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Begin a new recording session."""
        self._session = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._chunk   = 0
        self._total   = 0
        self._frames  = []
        self._labels  = []
        self.active   = True
        print(f"[Collector] Recording — session {self._session}  "
              f"(data → {self._dir}/)")

    def stop(self) -> None:
        """Flush remaining frames and end the session."""
        if self.active:
            self._flush()
            print(f"[Collector] Stopped.  {self._total} frames saved.")
        self.active = False

    # ------------------------------------------------------------------ #
    #  Per-frame call                                                      #
    # ------------------------------------------------------------------ #

    def record(self, frame: np.ndarray, keys: dict) -> None:
        """
        Append one (frame, [W,A,S,D]) sample for behavioural-cloning training.

        frame — full-size RGB uint8 array from grab_screen()
        keys  — {'W': bool, 'A': bool, 'S': bool, 'D': bool}
        """
        small = cv2.resize(frame, (INPUT_W, INPUT_H),
                           interpolation=cv2.INTER_AREA)
        label = np.array(
            [float(keys['W']), float(keys['A']),
             float(keys['S']), float(keys['D'])],
            dtype=np.float32,
        )
        self._frames.append(small)
        self._labels.append(label)
        if len(self._frames) >= CHUNK_SIZE:
            self._flush()

    def record_offset(self, frame: np.ndarray, offset: float) -> None:
        """
        Append one (frame, offset) sample for lane-offset regression training.

        frame  — full-size RGB uint8 array from grab_screen()
        offset — lane centre offset in pixels (from SteeringController).
                 Only call this when the lane detector has fresh bilateral
                 detections so every saved label is reliable.
        """
        small = cv2.resize(frame, (INPUT_W, INPUT_H),
                           interpolation=cv2.INTER_AREA)
        label = np.array([offset], dtype=np.float32)   # shape (1,)
        self._frames.append(small)
        self._labels.append(label)
        if len(self._frames) >= CHUNK_SIZE:
            self._flush()

    def record_obstacle(self, frame: np.ndarray) -> None:
        """
        Save one frame to training_data/obstacles/ labelled as obstacle-present.

        Called every frame while the user holds E (honk) during manual play.
        Builds a separate binary dataset (label=1.0) for obstacle detection
        training, independent of the steering offset dataset.
        """
        small = cv2.resize(frame, (INPUT_W, INPUT_H),
                           interpolation=cv2.INTER_AREA)
        self._obs_frames.append(small)
        if len(self._obs_frames) >= CHUNK_SIZE:
            self._flush_obstacles()

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def frame_count(self) -> int:
        """Total frames saved this session (includes unflushed buffer)."""
        return self._total + len(self._frames)

    @property
    def obstacle_frame_count(self) -> int:
        return self._obs_total + len(self._obs_frames)

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _flush(self) -> None:
        if not self._frames:
            return
        n    = len(self._frames)
        name = f'session_{self._session}_chunk_{self._chunk:03d}.npz'
        np.savez_compressed(
            self._dir / name,
            frames=np.stack(self._frames),   # (N, H, W, 3) uint8
            labels=np.stack(self._labels),   # (N, 4)        float32
        )
        self._total  += n
        self._chunk  += 1
        self._frames  = []
        self._labels  = []
        print(f"[Collector] {name}  ({n} frames, {self._total} total)")

    def _flush_obstacles(self) -> None:
        if not self._obs_frames:
            return
        n      = len(self._obs_frames)
        ts     = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        name   = f'obs_{ts}_chunk_{self._obs_chunk:03d}.npz'
        labels = np.ones((n, 1), dtype=np.float32)   # all obstacle-present
        np.savez_compressed(
            self._obs_dir / name,
            frames=np.stack(self._obs_frames),
            labels=labels,
        )
        self._obs_total  += n
        self._obs_chunk  += 1
        self._obs_frames  = []
        print(f"[Collector] {name}  ({n} obstacle frames, {self._obs_total} total)")
