"""
hand_tracker.py — MediaPipe Tasks-based hand detection & landmark extraction.

Uses the new MediaPipe Tasks Vision API (>= 0.10.21) which replaces the
legacy ``mp.solutions.hands`` interface.

Provides:
  • Hand detection with landmark drawing
  • 21-point landmark position extraction
  • Finger-up detection (thumb, index, middle, ring, pinky)
  • Distance measurement between any two landmarks
"""

import math
import os
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)


# Path to the bundled model (sits next to this file)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# MediaPipe hand-connection pairs for drawing (replicating the legacy drawing)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle  (fixed: 0→9 instead of 5→9)
    (0, 13), (13, 14), (14, 15), (15, 16), # ring    (fixed: 0→13)
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky   (fixed: 0→17)
    (5, 9), (9, 13), (13, 17),             # palm
]


class HandTracker:
    """Wraps MediaPipe Hand Landmarker (Tasks API) for real-time tracking."""

    def __init__(
        self,
        max_hands=1,
        detection_confidence=0.7,
        tracking_confidence=0.7,
    ):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=tracking_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        # Results cache
        self._result = None
        self.landmark_list = []

        # Tip IDs: thumb, index, middle, ring, pinky
        self.tip_ids = [4, 8, 12, 16, 20]

        # Frame timestamp counter (microseconds expected by VIDEO mode)
        self._ts = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_hands(self, frame, draw=True):
        """Detect hands and optionally draw landmarks on *frame*.

        Returns the annotated frame.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._ts += 33  # ~30 fps step
        self._result = self.landmarker.detect_for_video(mp_image, self._ts)

        if draw and self._result.hand_landmarks:
            h, w, _ = frame.shape
            for hand_lms in self._result.hand_landmarks:
                # Draw connections
                for c1, c2 in _HAND_CONNECTIONS:
                    x1, y1 = int(hand_lms[c1].x * w), int(hand_lms[c1].y * h)
                    x2, y2 = int(hand_lms[c2].x * w), int(hand_lms[c2].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw landmark dots
                for lm in hand_lms:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return frame

    def get_positions(self, frame, hand_index=0):
        """Return ``[(id, x, y), ...]`` for every landmark of *hand_index*."""
        self.landmark_list = []

        if (
            self._result
            and self._result.hand_landmarks
            and hand_index < len(self._result.hand_landmarks)
        ):
            hand = self._result.hand_landmarks[hand_index]
            h, w, _ = frame.shape
            for idx, lm in enumerate(hand):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmark_list.append((idx, cx, cy))

        return self.landmark_list

    def fingers_up(self):
        """Return ``[thumb, index, middle, ring, pinky]`` (1 = up, 0 = down).

        Requires ``get_positions()`` to have been called first.
        """
        if len(self.landmark_list) < 21:
            return [0, 0, 0, 0, 0]

        fingers = []

        # Thumb — compare x (works for right hand facing mirrored camera)
        if self.landmark_list[self.tip_ids[0]][1] < self.landmark_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other four fingers — tip above PIP joint (lower y = higher on screen)
        for i in range(1, 5):
            if self.landmark_list[self.tip_ids[i]][2] < self.landmark_list[self.tip_ids[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def find_distance(self, p1, p2, frame, draw=True):
        """Return ``(distance, frame, [x1,y1,x2,y2,cx,cy])``."""
        if len(self.landmark_list) < max(p1, p2) + 1:
            return 0, frame, []

        x1, y1 = self.landmark_list[p1][1], self.landmark_list[p1][2]
        x2, y2 = self.landmark_list[p2][1], self.landmark_list[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(frame, (cx, cy), 8, (0, 255, 0), cv2.FILLED)

        distance = math.hypot(x2 - x1, y2 - y1)
        return distance, frame, [x1, y1, x2, y2, cx, cy]
