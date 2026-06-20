"""
Screen capture that auto-detects the GTA V window position.

Uses FindWindowW + ClientToScreen + GetClientRect so the window can be
anywhere on screen — no manual positioning required.

Capture is done via mss (BitBlt-based) which is ~2-3x faster than
PIL.ImageGrab. A single mss.mss() context is created at module load
and reused every frame to avoid per-frame initialisation overhead.
"""

import ctypes
import ctypes.wintypes
from typing import Tuple

import mss
import numpy as np

_GTA_TITLE = "Grand Theft Auto V"

_sct  = mss.mss()
_bbox: Tuple[int, int, int, int] | None = None


def find_gta_window() -> Tuple[int, int, int, int]:
    """
    Locate the GTA V window and return its client-area bbox
    (left, top, right, bottom) in screen coordinates.
    """
    hwnd = ctypes.windll.user32.FindWindowW(None, _GTA_TITLE)
    if not hwnd:
        raise RuntimeError(
            f'Window "{_GTA_TITLE}" not found.\n'
            "  • Make sure GTA V is running.\n"
            "  • Set Display Mode to Windowed (not Borderless Windowed)."
        )
    pt = ctypes.wintypes.POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
    rc = ctypes.wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rc))
    bbox = (pt.x, pt.y, pt.x + rc.right, pt.y + rc.bottom)
    global _bbox
    _bbox = bbox
    return bbox


def get_gta_window_pos() -> Tuple[int, int] | None:
    """Return the current (x, y) top-left of the GTA V window, or None if not found."""
    hwnd = ctypes.windll.user32.FindWindowW(None, _GTA_TITLE)
    if not hwnd:
        return None
    rc = ctypes.wintypes.RECT()
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rc))
    return (rc.left, rc.top)


def move_gta_window(target_x: int, target_y: int) -> None:
    """
    Move the GTA V window so its top-left corner is at (target_x, target_y),
    then refresh the internal capture bbox.

    Does nothing (prints a warning) if the window is not found.
    """
    hwnd = ctypes.windll.user32.FindWindowW(None, _GTA_TITLE)
    if not hwnd:
        print("Warning: could not reposition GTA V window — window not found.")
        return

    # Preserve the current window size (only move, don't resize)
    wrc = ctypes.wintypes.RECT()
    ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(wrc))
    w = wrc.right  - wrc.left
    h = wrc.bottom - wrc.top

    ctypes.windll.user32.MoveWindow(hwnd, target_x, target_y, w, h, True)

    # Re-detect after the move so the capture bbox is correct
    find_gta_window()


def grab_screen() -> np.ndarray:
    """Return the GTA V client area as an (H, W, 3) RGB uint8 array."""
    if _bbox is None:
        raise RuntimeError("Call find_gta_window() before grab_screen().")
    left, top, right, bottom = _bbox
    monitor = {"left": left, "top": top, "width": right - left, "height": bottom - top}
    return np.array(_sct.grab(monitor), dtype=np.uint8)[:, :, 2::-1]
