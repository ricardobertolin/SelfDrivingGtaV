# Self-Driving GTA V

PIBITI proof-of-concept: lane detection via screen capture + OpenCV,
proportional steering controller, DirectInput keyboard output.

---

## Requirements

- Windows 10/11 (DirectInput keyboard control is Windows-only)
- Python 3.9+
- GTA V

---

## Setup

```bash
pip install -r requirements.txt
```

No additional drivers or libraries are required. All Windows API calls
go through the built-in `ctypes` module.

---

## GTA V configuration

| Setting | Value |
|---|---|
| Display mode | **Windowed** (not borderless) |
| Resolution | **800 × 600** |
| Window position | Anywhere — detected automatically at startup |
| Camera | Third-person or dashcam mod recommended |

At startup the script calls `FindWindowW("Grand Theft Auto V")` and then
`ClientToScreen` to locate the exact client area, so no manual positioning
is needed. The detected region is printed to the console before the
countdown begins so you can verify it.

If you change the game resolution, update `FRAME_WIDTH` / `FRAME_HEIGHT`
and the pixel-based constants (`ROI_VERTICES`, `ROI_TOP_Y`, `ROI_BOTTOM_Y`,
`TYPICAL_LANE_WIDTH`, `STEERING_DEADZONE`) in `config.py` to match.

---

## Running

1. Start GTA V and load into free-roam on a straight road.
2. Set Display Mode to **Windowed** at **800 × 600** — the window can be
   anywhere on your screen.
3. Run the script **from a regular terminal** (no admin required for capture;
   admin *is* required if SendInput is blocked by UAC on your system):

```bash
python main.py
```

4. The console prints the detected window region, then counts down 3 seconds.
   Alt-tab into GTA V during that time.
5. Press **Q** (globally) or **q** inside the OpenCV window to stop cleanly.

---

## Module overview

| File | Responsibility |
|---|---|
| `config.py` | All tunable constants in one place |
| `directkeys.py` | Windows `SendInput` wrappers (DirectInput scan codes) |
| `screen_capture.py` | PIL `ImageGrab` screen-region capture |
| `lane_detection.py` | `LaneDetector` class — edge detection, Hough, smoothing |
| `steering.py` | `SteeringController` — offset calculation, P-controller |
| `controller.py` | `KeyController` — stateful press/release, avoid key spam |
| `main.py` | Entry point, HUD rendering, main loop |

---

## How the controller works

```
screen capture
      │
      ▼
  grayscale → Gaussian blur → Canny edges → ROI mask → HoughLinesP
      │
      ▼
  classify lines by slope sign (negative = left lane, positive = right)
      │
      ▼
  fit one line per side, extend to ROI bounds
  update rolling buffer (last N frames), smooth by averaging
      │
      ▼
  lane_center = (left_x_bottom + right_x_bottom) / 2
  offset      = lane_center − image_center
      │
      ├── offset > +deadzone  → press D (steer right)
      ├── offset < −deadzone  → press A (steer left)
      └── |offset| ≤ deadzone → release A & D (straight)
      │
      └── always press W (accelerate)
```

Keys are only pressed/released when the desired state **changes**,
so there is no redundant SendInput spam every frame.

---

## Tuning guide

| Constant | Effect |
|---|---|
| `STEERING_DEADZONE` | Increase to steer less aggressively on slight curves |
| `SMOOTHING_WINDOW` | Increase for smoother but slower response; decrease for faster reaction |
| `MIN_LINES_FOR_STEERING` | Raise to require more edge evidence before steering |
| `TYPICAL_LANE_WIDTH` | Match to the pixel width of the lane in your camera view |
| `CANNY_LOW / CANNY_HIGH` | Tune for road colour / lighting conditions |
| `HOUGH_THRESHOLD` | Raise to reject shorter/noisier line candidates |

---

## Known limitations

- Lane detection relies on straight or gently curved roads.
- Performance is limited by `PIL.ImageGrab` (~10–20 FPS typical).
  Replace `screen_capture.py` with `mss` for 2–3× faster capture.
- The controller only handles throttle (W) and lateral steering (A/D);
  braking (S) is not implemented.
