import numpy as np

# --- Expected game client resolution (must match GTA V window size) ---
# The capture bbox is detected automatically from the window position at runtime.
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

# --- Edge detection ---
BLUR_KERNEL  = (5, 5)
CANNY_LOW    = 150
CANNY_HIGH   = 250

# --- Hough line transform ---
HOUGH_RHO          = 1
HOUGH_THETA        = np.pi / 180
HOUGH_THRESHOLD    = 100
HOUGH_MIN_LINE_LEN = 20
HOUGH_MAX_LINE_GAP = 15
MIN_LINES_FOR_STEERING = 5  # skip steering when fewer raw lines found

# --- Lane classification ---
SLOPE_MIN          = 0.3    # discard lines that are too horizontal
SLOPE_MAX          = 10.0   # discard lines that are too vertical
TYPICAL_LANE_WIDTH = 300    # px between lanes at ROI_BOTTOM_Y (for extrapolation)

# --- Temporal smoothing ---
SMOOTHING_WINDOW = 8  # rolling-average over this many frames

# --- Steering ---
STEERING_DEADZONE = 30   # px of offset before lateral steering is applied

# --- Speed control ---
# |offset| <= STEERING_DEADZONE          → accelerate (W)
# STEERING_DEADZONE < |offset| <= SPEED_BRAKE_THRESHOLD → coast (no W, no S)
# |offset| > SPEED_BRAKE_THRESHOLD       → brake (S)
SPEED_BRAKE_THRESHOLD = 80  # px of offset that triggers active braking

# --- DirectInput scan codes (used by GTA V) ---
KEY_W = 0x11
KEY_A = 0x1E
KEY_S = 0x1F
KEY_D = 0x20

# --- Kill switch (Windows Virtual Key code for Q) ---
KILL_VK = 0x51

# --- Display / HUD ---
SHOW_HUD          = True
HUD_COLOR         = (0, 255, 255)   # cyan text
LANE_COLOR        = (0, 255, 0)     # green lane lines
CENTER_LINE_COLOR = (0, 165, 255)   # orange center line
