import numpy as np

# --- Expected game client resolution (must match GTA V window size) ---
FRAME_WIDTH  = 800
FRAME_HEIGHT = 600

# --- Region of Interest (trapezoid vertices, x/y) ---
ROI_VERTICES = np.array([
    [10,  500],
    [10,  300],
    [300, 200],
    [500, 200],
    [800, 300],
    [800, 500],
], dtype=np.int32)

ROI_TOP_Y    = 200  # y where extended lines start (near horizon)
ROI_BOTTOM_Y = 500  # y where extended lines end   (near car)

# --- Edge detection (CANNY_LOW/HIGH used by obstacle detector) ---
BLUR_KERNEL = (5, 5)
CANNY_LOW   = 150
CANNY_HIGH  = 250

# --- Lane detection preprocessing ---
# CLAHE + adaptive Canny + HLS colour masking — no manual re-tuning for day/night.
CANNY_SIGMA      = 0.33   # auto-threshold: low=(1-σ)*median, high=(1+σ)*median
CLAHE_CLIP_LIMIT = 2.0    # CLAHE contrast limit  (raise for very dark scenes)
CLAHE_TILE_GRID  = (8, 8) # CLAHE tile size (smaller = more localised)

# HLS colour ranges for lane markings (H: 0-180, L/S: 0-255).
# White: strict L≥175 + low S to avoid picking up buildings/cars.
# Yellow: broad hue range, moderate L, any S — yellow paint stays saturated at night.
WHITE_LANE_LOW   = (  0, 175,   0)
WHITE_LANE_HIGH  = (180, 255,  60)
YELLOW_LANE_LOW  = ( 15,  50,  70)
YELLOW_LANE_HIGH = ( 45, 220, 255)

# Morphological closing after edge+colour merge (closes gaps without thickening)
MORPH_CLOSE_ITERS = 1

# --- Lane line fitting ---
LANE_MAX_JUMP_PX  = 100  # max x_bot change between frames before a detection is discarded
LANE_MIN_YSPAN_PX = 80   # minimum vertical span of segments (px); fits from a narrow
                          # y-band are unreliable because the polynomial is extrapolated
                          # over most of the ROI height — reject them.

# --- Hough line transform ---
HOUGH_RHO          = 1
HOUGH_THETA        = np.pi / 180
HOUGH_THRESHOLD    = 60   # lower than before → catches more shorter segments
HOUGH_MIN_LINE_LEN = 35   # longer minimum → rejects noisy 1-2 px blips
HOUGH_MAX_LINE_GAP = 25   # wider gap → better joins dashed lane markings
MIN_LINES_FOR_STEERING = 4   # steer with less evidence — missing a correction is worse than a small one

# --- Lane classification ---
SLOPE_MIN          = 0.3
SLOPE_MAX          = 10.0
TYPICAL_LANE_WIDTH = 300   # px between lanes at ROI_BOTTOM_Y (extrapolation fallback)

# --- Temporal smoothing ---
SMOOTHING_WINDOW = 8

# --- Steering ---
STEERING_DEADZONE = 25   # px of offset before lateral keys are pressed (react sooner)

# Pulsed steering: duty ≈ STEER_HOLD / (STEER_HOLD + STEER_SKIP)
# Set STEER_SKIP_FRAMES = 0 to disable pulsing (continuous hold).
STEER_HOLD_FRAMES = 4    # frames to hold the steering key per cycle
STEER_SKIP_FRAMES = 4    # frames to release between holds  (50% duty)

# --- Speed control ---
# |offset| ≤ STEERING_DEADZONE               → accelerate (W)
# STEERING_DEADZONE < |offset| ≤ BRAKE_THR   → coast
# |offset| > BRAKE_THR                        → pulsed brake (S tapped on/off)
SPEED_BRAKE_THRESHOLD = 40   # braking kicks in at small offsets to keep speed in check

# Pulsed braking — press S for BRAKE_HOLD_FRAMES then release for BRAKE_SKIP_FRAMES.
# GTA V reverses only if S is held continuously after the car stops; tapping avoids it.
BRAKE_HOLD_FRAMES  = 10   # frames to hold S per pulse
BRAKE_SKIP_FRAMES  = 4    # frames to press W between pulses (prevents reverse)

# Pulsed acceleration: W is not held continuously — tap it so the car
# cruises at a lower average speed, leaving more time for corrections.
ACCEL_HOLD_FRAMES  = 5    # frames to hold W per pulse
ACCEL_SKIP_FRAMES  = 3    # frames to release W between pulses

# Set False to prevent the S key from ever being pressed for speed control.
REVERSE_ENABLED = False

# --- DirectInput scan codes ---
KEY_W = 0x11
KEY_A = 0x1E
KEY_S = 0x1F
KEY_D = 0x20
KEY_E = 0x12   # horn / honk in GTA V

# --- Global VK codes ---
KILL_VK        = 0x51   # Q — kill switch
AI_CTRL_VK     = 0x43   # C — toggle full AI (lane-detection) control on/off
CAMERA_LOCK_VK = 0x4C   # L — toggle automatic camera-alignment correction
RECORD_VK      = 0x52   # R — toggle training-data recording (manual play only)
HONK_VK        = 0x45   # E — GTA V horn; also auto-starts recording (car-ahead situations)
ML_CTRL_VK     = 0x4D   # M — toggle ML model control (replaces lane detection)

# VK codes for detecting physical key presses during recording.
# These are the standard Windows virtual-key codes for letter keys
# (different from the DirectInput scan codes KEY_W/A/S/D above).
VK_W = 0x57
VK_A = 0x41
VK_S = 0x53
VK_D = 0x44

# --- ML / behavioural-cloning ---
ML_THRESHOLD        = 0.50               # sigmoid threshold for key-press model
ML_MODEL_PATH       = 'model.pth'        # DrivingCNN checkpoint (keys BC)
ML_OFFSET_MODEL_PATH = 'offset_model.pth'  # OffsetCNN checkpoint (recommended)

# --- Obstacle detection ---
OBSTACLE_DETECTION_ENABLED      = True
DANGER_ZONE                     = (300, 200, 500, 380)
OBSTACLE_EDGE_DENSITY_THRESHOLD = 0.15

# --- Camera alignment (car-body tracking) ---
# In GTA V third-person, the car rear is always visible in the lower-centre.
# The centroid of its edges is used as the alignment reference:
#   camera rotated left/right  → car shifts right/left  → horizontal correction
#   camera too high/low        → car shifts down/up     → vertical correction
# Tune CAR_ROI so the rectangle tightly wraps the car's visible rear in the overlay.
# If vertical correction is inverted on your setup, negate CAM_Y_GAIN.
CAR_ROI            = (150, 120, 650, 545)   # large search area — nearly the full frame
                                            # wide enough to catch the car even when the
                                            # camera is far off in any direction.
                                            # avoids x<150 (minimap) and y>545 (health bar).
CAR_TARGET_Y       = 390                    # centroid y in the ideal camera position
                                            # when centroid is above this → camera too low
                                            # when centroid is below this → camera too high
CAM_X_GAIN         = 0.80                   # mouse px per px of horizontal error
CAM_Y_GAIN         = 0.40                   # mouse px per px of vertical error
                                            # negate CAM_Y_GAIN if vertical correction is inverted
CAM_MIN_MOVE       = 5                      # minimum |mouse px| sent per correction tick
CAMERA_DEADZONE_PX = 15                     # ignore errors smaller than this
CAMERA_ALERT_PX    = 120                    # ROI turns red beyond this error (px)
CAR_MIN_CONTOUR_PX = 800                    # largest contour must exceed this area (px²)
                                            # to count as the car — filters background noise
OBSTACLE_ZONE_COLOR_CLEAR       = (0, 200,   0)
OBSTACLE_ZONE_COLOR_BLOCKED     = (0,   0, 255)

# --- Window layout ---
# GTA V window is moved here at startup so the overlay never overlaps it.
GTA_WINDOW_X     = 0     # px from left edge of primary monitor
GTA_WINDOW_Y     = 0     # px from top  edge of primary monitor
OVERLAY_WINDOW_X = 820   # where the OpenCV debug window is placed (just right of GTA V)
OVERLAY_WINDOW_Y = 0

# --- Display / HUD ---
SHOW_HUD           = True
HUD_COLOR          = (0, 255, 255)   # cyan  — normal HUD text
LANE_COLOR         = (0, 255, 0)     # green — freshly detected lane
LANE_STALE_COLOR   = (0, 140, 255)   # orange — lane from smoothing buffer / extrapolated
CENTER_LINE_COLOR  = (255, 100, 0)   # blue  — road centre line
LANE_CURVE_POINTS  = 30              # sample points along each polynomial curve
