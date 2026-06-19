from directkeys import PressKey, ReleaseKey


class KeyController:
    """
    Tracks which DirectInput keys are currently held and only sends
    press/release events when the desired state changes, avoiding
    redundant SendInput calls every frame.
    """

    def __init__(self) -> None:
        self._held: set = set()

    def press(self, scan_code: int) -> None:
        if scan_code not in self._held:
            PressKey(scan_code)
            self._held.add(scan_code)

    def release(self, scan_code: int) -> None:
        if scan_code in self._held:
            ReleaseKey(scan_code)
            self._held.discard(scan_code)

    def release_all(self) -> None:
        for key in list(self._held):
            ReleaseKey(key)
        self._held.clear()
