
https://github.com/user-attachments/assets/60bd818a-e086-4823-ac8b-4766ee370719

# Self-Driving GTA V

PIBITI university research project — autonomous driving in GTA V using screen capture, computer vision, and a trained neural network.

---

## How it works

Two driving modes are available and can be switched at runtime:

### Lane-detection AI (`C` key)
Classic computer-vision pipeline that runs every frame:

```
screen capture (mss)
      │
      ▼
  HLS colour space → CLAHE contrast → adaptive Canny → colour mask (white + yellow)
      │
      ▼
  ROI trapezoid mask → HoughLinesP
      │
      ▼
  classify segments by slope sign AND x-position (left / right lanes)
  IQR outlier removal → length-weighted polynomial fit (degree 2)
  temporal smoothing buffer (last N frames) → extrapolate missing side
      │
      ▼
  lane_center = (left_x_bottom + right_x_bottom) / 2
  offset      = lane_center − image_center
      │
      ├── |offset| ≤ STEERING_DEADZONE     → W pulsed (cruise)
      ├── DEADZONE < |offset| ≤ BRAKE_THR  → coast
      └── |offset| > BRAKE_THR             → pulsed brake (S/W alternating)
      │
      └── offset > 0  → pulse D    |    offset < 0  → pulse A
```

### ML offset model (`M` key)
A CNN (`OffsetCNN`) trained on your own recorded drives predicts the same lane-centre offset directly from raw pixels — no Hough lines, no colour masks. The predicted offset feeds into the same steering and speed controller as the lane-detection mode.

---

## Controls

| Key | Action |
|-----|--------|
| `C` | Toggle lane-detection AI on / off |
| `M` | Toggle ML model on / off (requires trained `offset_model.pth`) |
| `R` | Toggle training-data recording — only during manual play |
| `E` | GTA V horn; automatically records obstacle frames while held |
| `L` | Toggle camera-alignment correction |
| `Q` | Kill switch — stops the script |

> **C and M are mutually exclusive.** Enabling one disables the other.
> Recording is blocked while either AI mode is active.

---

## Requirements

- Windows 10/11 (DirectInput keyboard control is Windows-only)
- Python 3.9+
- GTA V

```bash
pip install -r requirements.txt
```

PyTorch is required only for ML mode. If not installed, lane-detection AI still works and the script starts without errors.

---

## GTA V configuration

| Setting | Value |
|---------|-------|
| Display mode | **Windowed** (not borderless) |
| Resolution | **800 × 600** |
| Camera | Third-person |

At startup the script locates the GTA V window automatically via `FindWindowW`, moves it to `(GTA_WINDOW_X, GTA_WINDOW_Y)`, and prints the detected capture region to the console before the countdown.

---

## Running

1. Start GTA V and load into free-roam on a paved road.
2. Set Display Mode → **Windowed** at **800 × 600**.
3. Run:

```bash
python main.py
```

4. The console counts down 3 seconds — alt-tab into GTA V.
5. Press **C** for lane-detection AI or **M** for ML model.

---

## Training your own model

> **The trained model files (`offset_model.pth`, `model.pth`) are not included in this repository.** You must record your own driving data and train locally.

### 1 — Record training data

Drive manually in GTA V and press **R** to start recording. The system saves one frame + lane-offset label per frame **only when both lanes are confidently detected**, so every saved sample is clean.

Obstacle situations: hold **E** (honk) when a vehicle is in front — frames are saved separately to `training_data/obstacles/` for future obstacle-classifier training.

Recording is saved automatically in chunks to `training_data/`. Stop with **R** again.

### 2 — Train

```bash
python train_model.py
```

The script loads all `.npz` files in `training_data/`, discards chunks with fewer than 50 frames, and trains `OffsetCNN` (recommended) or `DrivingCNN` depending on the label format. The best checkpoint is saved as `offset_model.pth`.

Training output shows MAE in pixels — below ~30 px is a reasonable starting point.

### 3 — Use

Press **M** in `main.py`. The model is loaded automatically at startup if `offset_model.pth` exists.

---

## Module overview

| File | Responsibility |
|------|----------------|
| `config.py` | All tunable constants |
| `screen_capture.py` | `mss`-based screen region capture + GTA V window detection |
| `lane_detection.py` | `LaneDetector` — HLS preprocessing, Hough, polynomial fitting, temporal smoothing |
| `steering.py` | `SteeringController` — offset → action / speed decision |
| `obstacle_detection.py` | `ObstacleDetector` — Canny edge density in forward danger zone |
| `camera_controller.py` | `CameraController` — 1-D projection centroid, mouse correction |
| `controller.py` | `KeyController` — stateful DirectInput press / release |
| `model.py` | `OffsetCNN` (offset regression) and `DrivingCNN` (behavioural cloning) |
| `data_collector.py` | `DataCollector` — frame + label recording, chunked `.npz` storage |
| `train_model.py` | Offline training — auto-detects label type, saves best checkpoint |
| `main.py` | Entry point, main loop, HUD, key handling |

---

## Tuning guide

| Constant | Effect |
|----------|--------|
| `STEERING_DEADZONE` | Smaller → reacts to smaller offsets (more sensitive) |
| `STEER_HOLD_FRAMES` / `STEER_SKIP_FRAMES` | Steering duty cycle — raise hold or lower skip for more steering authority |
| `ACCEL_HOLD_FRAMES` / `ACCEL_SKIP_FRAMES` | W pulse rate — lower hold or raise skip to cruise slower |
| `BRAKE_HOLD_FRAMES` / `BRAKE_SKIP_FRAMES` | Braking duty cycle — raise hold for stronger deceleration |
| `SPEED_BRAKE_THRESHOLD` | Offset (px) at which braking starts — lower to brake earlier on curves |
| `MIN_LINES_FOR_STEERING` | Minimum Hough segments before lane-detection steering activates |
| `SMOOTHING_WINDOW` | Frames of temporal averaging — higher = smoother but slower to react |
| `LANE_MIN_YSPAN_PX` | Reject fits where detected segments span less than this vertical range |

---

## Known limitations

- Lane detection works best on paved straight or gently curved roads with visible markings.
- The ML model quality depends entirely on the amount and quality of recorded data — more varied recordings produce better generalisation.
- Obstacle detection uses a simple Canny edge-density heuristic; it can trigger on road markings or distant scenery in some lighting conditions.
- DirectInput keyboard output is Windows-only.
