import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# ==============================
# USER SETTINGS (adjust here)
# ==============================

# Distances are normalized (0..1).
LEFT_TRIGGER_DISTANCE = 0.030    # Thumb–index tip threshold (LEFT)
RIGHT_TRIGGER_DISTANCE = 0.030   # Thumb–middle tip threshold (RIGHT)

# Optional hand-size scaling to keep behavior consistent across distances
USE_HAND_SIZE_SCALING = True     # True/False
SCALE_FACTOR = 0.8               # Typical 0.6–1.0; multiplies wrist->index_mcp reference
MIN_FLOOR = 0.010                # Safety floor for dynamic thresholds

# Stability (frames to confirm press/release)
TOUCH_FRAMES_REQUIRED = 2        # 1–3: higher = steadier, slower
RELEASE_FRAMES_REQUIRED = 2

# Require index up to allow pointer/clicks
INDEX_MUST_BE_UP_FOR_CLICKS = True

# Cursor smoothing (higher = smoother, slower)
SMOOTHING = 10

# Camera capture settings
CAM_WIDTH, CAM_HEIGHT = 640, 480
REQUEST_FPS = 60

# Line thickness for on-screen guides
LINE_THICKNESS = 3

# ==============================
# IMPLEMENTATION
# ==============================

class HandMouseController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.screen_width, self.screen_height = pyautogui.size()

        self.prev_x, self.prev_y = 0, 0

        # True drag states
        self.left_down = False
        self.right_down = False

        # Debounce counters
        self.left_touch_frames = 0
        self.left_release_frames = 0
        self.right_touch_frames = 0
        self.right_release_frames = 0

        # Debug values
        self.last_index_up = False
        self.last_d_left = 0.0     # thumb-index tip distance (LEFT)
        self.last_d_right = 0.0    # thumb-middle tip distance (RIGHT)
        self.last_thr_left = 0.0
        self.last_thr_right = 0.0

        pyautogui.FAILSAFE = False

        # Landmark indices
        self.IDX = {
            'WRIST': 0,
            'THUMB_TIP': 4,
            'INDEX_MCP': 5,
            'INDEX_PIP': 6,
            'INDEX_TIP': 8,
            'MIDDLE_TIP': 12,
        }

    def get_points(self, lm_list):
        if len(lm_list) < 21:
            raise ValueError(f"Expected 21 landmarks, got {len(lm_list)}")
        g = self.IDX
        return {
            'wrist': lm_list[g['WRIST']],
            'index_mcp': lm_list[g['INDEX_MCP']],
            'index_pip': lm_list[g['INDEX_PIP']],
            'index_tip': lm_list[g['INDEX_TIP']],
            'middle_tip': lm_list[g['MIDDLE_TIP']],
            'thumb_tip': lm_list[g['THUMB_TIP']],
        }

    @staticmethod
    def dist(p1, p2):
        return float(np.hypot(p1.x - p2.x, p1.y - p2.y))

    @staticmethod
    def is_index_up(tip, pip):
        # MediaPipe normalized y increases downward; tip above pip means "up"
        return tip.y < pip.y

    def smooth_move(self, x, y):
        sx = self.prev_x + (x - self.prev_x) / SMOOTHING
        sy = self.prev_y + (y - self.prev_y) / SMOOTHING
        self.prev_x, self.prev_y = sx, sy
        return int(sx), int(sy)

    def detect_and_update(self, pts):
        # Distances
        d_thumb_index = self.dist(pts['thumb_tip'], pts['index_tip'])    # LEFT
        d_thumb_middle = self.dist(pts['thumb_tip'], pts['middle_tip'])  # RIGHT
        self.last_d_left = d_thumb_index
        self.last_d_right = d_thumb_middle

        # Thresholds (scaled if enabled)
        if USE_HAND_SIZE_SCALING:
            ref_len = self.dist(pts['wrist'], pts['index_mcp'])
            dyn = max(MIN_FLOOR, ref_len * SCALE_FACTOR)
            thr_left = max(LEFT_TRIGGER_DISTANCE, dyn)
            thr_right = max(RIGHT_TRIGGER_DISTANCE, dyn)
        else:
            thr_left = LEFT_TRIGGER_DISTANCE
            thr_right = RIGHT_TRIGGER_DISTANCE

        self.last_thr_left = thr_left
        self.last_thr_right = thr_right

        # Touch decisions
        left_touch_now = d_thumb_index < thr_left
        right_touch_now = d_thumb_middle < thr_right

        # Index up gating
        index_up = self.is_index_up(pts['index_tip'], pts['index_pip'])
        self.last_index_up = index_up
        clicks_ok = (index_up or not INDEX_MUST_BE_UP_FOR_CLICKS)

        # Debounce for left
        left_confirm = False
        left_release = False
        if left_touch_now:
            self.left_touch_frames += 1
            self.left_release_frames = 0
            if self.left_touch_frames >= TOUCH_FRAMES_REQUIRED:
                left_confirm = True
        else:
            self.left_release_frames += 1
            if self.left_release_frames >= RELEASE_FRAMES_REQUIRED:
                self.left_touch_frames = 0
                left_release = True

        # Debounce for right
        right_confirm = False
        right_release = False
        if right_touch_now:
            self.right_touch_frames += 1
            self.right_release_frames = 0
            if self.right_touch_frames >= TOUCH_FRAMES_REQUIRED:
                right_confirm = True
        else:
            self.right_release_frames += 1
            if self.right_release_frames >= RELEASE_FRAMES_REQUIRED:
                self.right_touch_frames = 0
                right_release = True

        return clicks_ok, left_confirm, left_release, right_confirm, right_release

    def handle_actions(self, clicks_ok, left_confirm, left_release, right_confirm, right_release, pointer_pos):
        # Pointer movement when index is up
        if self.last_index_up:
            sx, sy = int(pointer_pos.x * self.screen_width), int(pointer_pos.y * self.screen_height)
            mx, my = self.smooth_move(sx, sy)
            pyautogui.moveTo(mx, my)

        # LEFT drag (mouseDown on confirm, mouseUp on release)
        if clicks_ok and left_confirm and not self.left_down:
            pyautogui.mouseDown(button='left')
            self.left_down = True
            # Ensure mutual exclusivity
            if self.right_down:
                pyautogui.mouseUp(button='right')
                self.right_down = False
            print("Left Down")
        if self.left_down and left_release:
            pyautogui.mouseUp(button='left')
            self.left_down = False
            print("Left Up")

        # RIGHT drag
        if clicks_ok and right_confirm and not self.right_down:
            pyautogui.mouseDown(button='right')
            self.right_down = True
            if self.left_down:
                pyautogui.mouseUp(button='left')
                self.left_down = False
            print("Right Down")
        if self.right_down and right_release:
            pyautogui.mouseUp(button='right')
            self.right_down = False
            print("Right Up")

    def draw_overlay(self, frame, hand_landmarks, pts):
        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Status text
        cv2.putText(frame, f"Pointing: {self.last_index_up}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if self.last_index_up else (0,0,255), 2)
        cv2.putText(frame, f"Left Down: {self.left_down}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if self.left_down else (0,0,255), 2)
        cv2.putText(frame, f"Right Down: {self.right_down}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if self.right_down else (0,0,255), 2)

        # Distances vs thresholds
        cv2.putText(frame, f"LEFT  d(t-i)={self.last_d_left:.3f}  < thr {self.last_thr_left:.3f}", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)
        cv2.putText(frame, f"RIGHT d(t-m)={self.last_d_right:.3f} < thr {self.last_thr_right:.3f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2)

        # Draw connector lines and fingertip dots
        h, w = frame.shape[:2]
        def pt_to_xy(p): return int(p.x * w), int(p.y * h)

        thumb_xy = pt_to_xy(pts['thumb_tip'])
        index_xy = pt_to_xy(pts['index_tip'])
        middle_xy = pt_to_xy(pts['middle_tip'])

        # Line colors: normal vs active
        left_color = (0, 200, 0) if not self.left_down else (0, 255, 0)   # green; brighter when active
        right_color = (200, 0, 0) if not self.right_down else (255, 255, 0) # blue-ish to cyan when active

        # Draw lines
        cv2.line(frame, thumb_xy, index_xy, left_color, LINE_THICKNESS)
        cv2.line(frame, thumb_xy, middle_xy, right_color, LINE_THICKNESS)

        # Dots
        cv2.circle(frame, thumb_xy, 6, (0,255,255), -1)   # thumb: yellow
        cv2.circle(frame, index_xy, 6, (0,255,0), -1)     # index: green
        cv2.circle(frame, middle_xy, 6, (255,0,255), -1)  # middle: purple

        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, REQUEST_FPS)

        print("Hand Mouse Controller Started!")
        print("- Move: index finger up (cursor follows)")
        print("- Left drag: thumb tip close to index tip (hold to drag)")
        print("- Right drag: thumb tip close to middle tip (hold to drag)")
        print("- Press 'q' in the video window to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                lm_list = hand.landmark
                try:
                    pts = self.get_points(lm_list)
                    clicks_ok, l_conf, l_rel, r_conf, r_rel = self.detect_and_update(pts)
                    self.handle_actions(clicks_ok, l_conf, l_rel, r_conf, r_rel, pts['index_tip'])
                    frame = self.draw_overlay(frame, hand, pts)
                except (TypeError, ValueError) as e:
                    print(f"[Landmark error] {e}")
            else:
                cv2.putText(frame, "No hand detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("Gesture Mouse", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Safety: release any held buttons
        if self.left_down:
            pyautogui.mouseUp(button='left')
        if self.right_down:
            pyautogui.mouseUp(button='right')

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        HandMouseController().run()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"An error occurred: {e}")
