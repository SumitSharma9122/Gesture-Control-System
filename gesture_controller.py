"""
gesture_controller.py — Gesture recognition → OS mouse actions.

Translates finger states and distances into:
  • Smooth cursor movement
  • Left / right / double click (with cooldown)
  • Scroll up / down
  • Drag & drop
"""

import time
import numpy as np
import pyautogui

# Safety settings — disable fail-safe corner and remove inter-command pause
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


class GestureController:
    """Converts hand-gesture data into mouse actions."""

    def __init__(self, smoothing=7, click_cooldown=0.4):
        # Screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()

        # Smoothing state for cursor
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.smoothing = smoothing

        # Click cooldown (seconds) to prevent accidental double-fires
        self.click_cooldown = click_cooldown
        self._last_click_time = 0.0

        # Drag state
        self._dragging = False

        # Reduction frame (ignore edges to improve accuracy)
        self.frame_margin = 80  # pixels to ignore around the webcam frame

    # ------------------------------------------------------------------
    # Cursor movement
    # ------------------------------------------------------------------

    def move_cursor(self, index_x, index_y, frame_w, frame_h):
        """Move the OS cursor based on the index-finger position.

        Uses linear interpolation + exponential smoothing to avoid jitter.
        """
        # Map webcam coordinates → screen coordinates
        target_x = np.interp(
            index_x,
            (self.frame_margin, frame_w - self.frame_margin),
            (0, self.screen_w),
        )
        target_y = np.interp(
            index_y,
            (self.frame_margin, frame_h - self.frame_margin),
            (0, self.screen_h),
        )

        # Exponential moving average for smoothness
        cur_x = self.prev_x + (target_x - self.prev_x) / self.smoothing
        cur_y = self.prev_y + (target_y - self.prev_y) / self.smoothing

        # Clamp to screen bounds
        cur_x = max(0, min(self.screen_w - 1, cur_x))
        cur_y = max(0, min(self.screen_h - 1, cur_y))

        pyautogui.moveTo(cur_x, cur_y)
        self.prev_x, self.prev_y = cur_x, cur_y

    # ------------------------------------------------------------------
    # Click actions
    # ------------------------------------------------------------------

    def _can_click(self):
        now = time.time()
        if now - self._last_click_time >= self.click_cooldown:
            self._last_click_time = now
            return True
        return False

    def left_click(self):
        """Perform a left click if cooldown has elapsed."""
        if self._can_click():
            pyautogui.click()

    def right_click(self):
        """Perform a right click if cooldown has elapsed."""
        if self._can_click():
            pyautogui.rightClick()

    def double_click(self):
        """Perform a double click if cooldown has elapsed."""
        if self._can_click():
            pyautogui.doubleClick()

    # ------------------------------------------------------------------
    # Scroll
    # ------------------------------------------------------------------

    @staticmethod
    def scroll_up(amount=25):
        pyautogui.scroll(amount)

    @staticmethod
    def scroll_down(amount=25):
        pyautogui.scroll(-amount)

    # ------------------------------------------------------------------
    # Drag & Drop
    # ------------------------------------------------------------------

    def start_drag(self):
        """Begin dragging (hold left button)."""
        if not self._dragging:
            pyautogui.mouseDown()
            self._dragging = True

    def stop_drag(self):
        """Release the left button to finish dragging."""
        if self._dragging:
            pyautogui.mouseUp()
            self._dragging = False

    @property
    def is_dragging(self):
        return self._dragging
