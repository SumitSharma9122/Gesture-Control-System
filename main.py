"""
main.py — Gesture Controlled Virtual Mouse

Captures webcam video, detects the hand, interprets finger gestures,
and drives the OS cursor in real time.

Gesture table (simplified & reliable):
  Index only up       → Move cursor
  Two fingers pinch   → Left click  (index + middle tips close)
  Two fingers apart   → Scroll      (move hand up / down)
  Three fingers up    → Right click  (index+middle+ring, pinch index-middle)
  Fist (no fingers)   → Drag (hold)
  All five up         → Idle / stop

Press 'q' to quit.
"""

import time
import cv2
from hand_tracker import HandTracker
from gesture_controller import GestureController


# ── Configuration ────────────────────────────────────────────────────
CAM_INDEX = 0
CAM_WIDTH = 640
CAM_HEIGHT = 480
PINCH_THRESHOLD = 50        # pixels – distance to count as "pinch"
SMOOTHING = 7
CLICK_COOLDOWN = 0.5        # seconds between clicks
SCROLL_AMOUNT = 20          # lines per scroll tick
SCROLL_COOLDOWN = 0.15      # seconds between scroll ticks


def main():
    # ── Initialise camera ────────────────────────────────────────────
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Try changing CAM_INDEX to 1.")
        return

    tracker = HandTracker(max_hands=1, detection_confidence=0.7, tracking_confidence=0.7)
    controller = GestureController(smoothing=SMOOTHING, click_cooldown=CLICK_COOLDOWN)

    prev_time = time.time()
    last_scroll_time = 0.0          # cooldown timer for scroll

    print("[INFO] Gesture Virtual Mouse started.  Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            continue

        # Mirror the feed so it feels natural
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # ── Detect hand & landmarks ─────────────────────────────────
        frame = tracker.find_hands(frame)
        positions = tracker.get_positions(frame)

        if positions:
            fingers = tracker.fingers_up()
            total_up = sum(fingers)

            # Key landmark positions
            index_x, index_y = positions[8][1], positions[8][2]
            middle_x, middle_y = positions[12][1], positions[12][2]

            # Distance between index and middle fingertips
            dist_im, frame, _ = tracker.find_distance(8, 12, frame, draw=False)

            # Also show current gesture name on screen for debugging
            gesture_name = f"Fingers: {fingers}"
            cv2.putText(frame, gesture_name, (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            # ─────────────────────────────────────────────────────────
            # GESTURE 1: Move cursor — only INDEX finger up
            # ─────────────────────────────────────────────────────────
            if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                controller.stop_drag()
                scroll_prev_y = None
                controller.move_cursor(index_x, index_y, w, h)
                cv2.circle(frame, (index_x, index_y), 12, (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, "MOVE", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # ─────────────────────────────────────────────────────────
            # GESTURE 2: Two fingers up (index + middle)
            #   → Pinch = LEFT CLICK
            #   → Apart = SCROLL (based on hand movement)
            # ─────────────────────────────────────────────────────────
            elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
                controller.stop_drag()

                if dist_im < PINCH_THRESHOLD:
                    # ── Two fingers PINCHED → LEFT CLICK ─────────
                    controller.left_click()
                    cv2.circle(frame, ((index_x + middle_x) // 2, (index_y + middle_y) // 2),
                               15, (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, "LEFT CLICK", (20, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    # ── Two fingers APART → SCROLL ───────────────
                    # Position-based: hand above line → up, below → down
                    # Line at 35% height so there's plenty of room below for scroll-down
                    mid_y = (index_y + middle_y) // 2
                    center_y = int(h * 0.35)
                    deadzone = 20
                    now = time.time()

                    if now - last_scroll_time >= SCROLL_COOLDOWN:
                        if mid_y < center_y - deadzone:
                            controller.scroll_up(SCROLL_AMOUNT)
                            cv2.putText(frame, "SCROLL UP", (20, 130),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                            last_scroll_time = now
                        elif mid_y > center_y + deadzone:
                            controller.scroll_down(SCROLL_AMOUNT)
                            cv2.putText(frame, "SCROLL DOWN", (20, 130),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
                            last_scroll_time = now
                        else:
                            cv2.putText(frame, "SCROLL READY", (20, 130),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 180), 2)

                    # Draw guide: center line + zone labels
                    cv2.line(frame, (0, center_y), (w, center_y), (100, 100, 100), 1)
                    cv2.putText(frame, "^ UP ^", (w - 120, center_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    cv2.putText(frame, "v DOWN v", (w - 130, center_y + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
                    cv2.circle(frame, (index_x, index_y), 10, (255, 255, 0), cv2.FILLED)
                    cv2.circle(frame, (middle_x, middle_y), 10, (255, 255, 0), cv2.FILLED)

            # ─────────────────────────────────────────────────────────
            # GESTURE 3: Three fingers up → RIGHT CLICK
            # ─────────────────────────────────────────────────────────
            elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0:
                controller.stop_drag()
                controller.right_click()
                cv2.putText(frame, "RIGHT CLICK", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # ─────────────────────────────────────────────────────────
            # GESTURE 4: Four fingers up → DOUBLE CLICK
            # ─────────────────────────────────────────────────────────
            elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1 and fingers[0] == 0:
                controller.stop_drag()
                controller.double_click()
                cv2.putText(frame, "DOUBLE CLICK", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            # ─────────────────────────────────────────────────────────
            # GESTURE 5: Fist (no fingers) → DRAG
            # ─────────────────────────────────────────────────────────
            elif total_up == 0:
                controller.start_drag()
                controller.move_cursor(index_x, index_y, w, h)
                cv2.putText(frame, "DRAGGING", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)

            # ─────────────────────────────────────────────────────────
            # GESTURE 6: All five up → IDLE
            # ─────────────────────────────────────────────────────────
            elif total_up == 5:
                controller.stop_drag()
                cv2.putText(frame, "IDLE", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

            else:
                controller.stop_drag()

        # ── FPS overlay ─────────────────────────────────────────────
        cur_time = time.time()
        fps = 1 / (cur_time - prev_time + 1e-9)
        prev_time = cur_time
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ── Show frame ──────────────────────────────────────────────
        cv2.imshow("Gesture Virtual Mouse", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Cleanup ─────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Gesture Virtual Mouse stopped.")


if __name__ == "__main__":
    main()
