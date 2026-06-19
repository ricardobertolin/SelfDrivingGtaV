"""
Screen capture that auto-detects the GTA V window position.

Uses FindWindowW + ClientToScreen + GetClientRect so the window can be
anywhere on screen — no manual positioning required.
"""

import ctypes
import ctypes.wintypes
from typing import Tuple

import numpy as np
from PIL import ImageGrab

_GTA_TITLE = "Grand Theft Auto V"

# Populated once by find_gta_window(); used by every grab_screen() call.
_bbox: Tuple[int, int, int, int] | None = None


def find_gta_window() -> Tuple[int, int, int, int]:
    """
    Locate the GTA V window and return its client-area bbox
    (left, top, right, bottom) in screen coordinates.

    Raises RuntimeError if the window is not found.
    """
    hwnd = ctypes.windll.user32.FindWindowW(None, _GTA_TITLE)
    if not hwnd:
        raise RuntimeError(
            f'Window "{_GTA_TITLE}" not found.\n'
            "  • Make sure GTA V is running.\n"
            "  • Set Display Mode to Windowed (not Borderless Windowed)."
        )

    # Client-area top-left in screen coordinates
    pt = ctypes.wintypes.POINT(0, 0)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))

    # Client-area dimensions (always origin-relative, so right == width)
    rc = ctypes.wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rc))

    bbox = (pt.x, pt.y, pt.x + rc.right, pt.y + rc.bottom)

    global _bbox
    _bbox = bbox
    return bbox


def grab_screen() -> np.ndarray:
    """Return the GTA V client area as an (H, W, 3) RGB uint8 array."""
    if _bbox is None:
        raise RuntimeError(
            "Call find_gta_window() before grab_screen()."
        )
    return np.array(ImageGrab.grab(bbox=_bbox))
