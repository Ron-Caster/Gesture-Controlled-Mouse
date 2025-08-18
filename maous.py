import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

class HandMouseController:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Screen and camera
        self.screen_width, self.screen_height = pyautogui.size()
        self.cam_width, self.cam_height = 640, 480
        self.request_fps = 60

        # Pointer smoothing
        self.smoothing = 10
        self.prev_x, self.prev_y = 0, 0

        # Independent cooldowns
        self.left_cooldown = 0.2
        self.right_cooldown = 0.2
        self.last_left_time = 0.0
        self.last_right_time = 0.0

        # Debounce/hysteresis
        self.touch_frames_required = 2     # frames to confirm click
        self.release_frames_required = 2   # frames to confirm release
        self.left_touch_frames = 0
        self.right_touch_frames = 0
        self.left_release_frames = 0
        self.right_release_frames = 0

        # Click states
        self.left_click_active = False
        self.right_click_active = False

        # Dynamic threshold base (scaled by hand size)
        self.min_touch_threshold = 0.03
        self.scale_factor = 0.8
        self.index_must_be_up_for_clicks = True  # optional guard

        # Debug info
        self.last_d_left = 0.0   # thumb-index distance (left click)
        self.last_d_right = 0.0  # thumb-middle distance (right click)
        self.last_dyn_thresh = 0.0
        self.last_index_up = False

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

    def is_index_up(self, tip, pip):
        # MediaPipe y increases downward; tip above pip => tip.y < pip.y
        return tip.y < pip.y

    def dist(self, p1, p2):
        return float(np.hypot(p1.x - p2.x, p1.y - p2.y))

    def detect_gestures(self, pts):
        # Hand size reference: wrist to index_mcp
        ref_len = self.dist(pts['wrist'], pts['index_mcp'])
        dyn_thresh = max(self.min_touch_threshold, ref_len * self.scale_factor)
        self.last_dyn_thresh = dyn_thresh

        # Pointer gating
        index_up = self.is_index_up(pts['index_tip'], pts['index_pip'])
        self.last_index_up = index_up

        # MAPPING REQUIRED:
        # Left click = thumb + index tip
        # Right click = thumb + middle tip
        d_thumb_index = self.dist(pts['thumb_tip'], pts['index_tip'])   # LEFT
        d_thumb_middle = self.dist(pts['thumb_tip'], pts['middle_tip']) # RIGHT

        self.last_d_left = d_thumb_index
        self.last_d_right = d_thumb_middle

        left_touch_now = d_thumb_index < dyn_thresh
        right_touch_now = d_thumb_middle < dyn_thresh

        # Debounce left
        left_click = False
        if left_touch_now:
            self.left_touch_frames += 1
            self.left_release_frames = 0
            if self.left_touch_frames >= self.touch_frames_required:
                left_click = True
        else:
            self.left_release_frames += 1
            if self.left_release_frames >= self.release_frames_required:
                self.left_touch_frames = 0

        # Debounce right
        right_click = False
        if right_touch_now:
            self.right_touch_frames += 1
            self.right_release_frames = 0
            if self.right_touch_frames >= self.touch_frames_required:
                right_click = True
        else:
            self.right_release_frames += 1
            if self.right_release_frames >= self.release_frames_required:
                self.right_touch_frames = 0

        return {
            'pointing': index_up,
            'left_click': left_click,
            'right_click': right_click,
            'pointer_pos': pts['index_tip']
        }

    def smooth_move(self, x, y):
        sx = self.prev_x + (x - self.prev_x) / self.smoothing
        sy = self.prev_y + (y - self.prev_y) / self.smoothing
        self.prev_x, self.prev_y = sx, sy
        return int(sx), int(sy)

    def handle_actions(self, gestures):
        now = time.time()

        # Pointer move only when index is up
        if gestures['pointing']:
            pos = gestures['pointer_pos']
            sx, sy = int(pos.x * self.screen_width), int(pos.y * self.screen_height)
            mx, my = self.smooth_move(sx, sy)
            pyautogui.moveTo(mx, my)

        # Optional: also require index up to allow clicks at all
        clicks_ok = (gestures['pointing'] or not self.index_must_be_up_for_clicks)

        # Left click (thumb + index)
        if clicks_ok and gestures['left_click'] and not self.left_click_active:
            if (now - self.last_left_time) > self.left_cooldown:
                pyautogui.click(button='left')
                self.left_click_active = True
                self.last_left_time = now
                print("Left Click")
        elif not gestures['left_click']:
            self.left_click_active = False

        # Right click (thumb + middle)
        if clicks_ok and gestures['right_click'] and not self.right_click_active:
            if (now - self.last_right_time) > self.right_cooldown:
                pyautogui.click(button='right')
                self.right_click_active = True
                self.last_right_time = now
                print("Right Click")
        elif not gestures['right_click']:
            self.right_click_active = False

    def draw_overlay(self, img, hand_landmarks, gestures, pts):
        self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Status
        cv2.putText(img, f"Pointing: {gestures['pointing']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if gestures['pointing'] else (0,0,255), 2)
        cv2.putText(img, f"Left Click: {gestures['left_click']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if gestures['left_click'] else (0,0,255), 2)
        cv2.putText(img, f"Right Click: {gestures['right_click']}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if gestures['right_click'] else (0,0,255), 2)

        # Debug distances
        cv2.putText(img, f"d(thumb-index)= {self.last_d_left:.3f}  [LEFT]", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.putText(img, f"d(thumb-middle)= {self.last_d_right:.3f} [RIGHT]", (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        cv2.putText(img, f"dyn_thresh: {self.last_dyn_thresh:.3f}", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,200), 2)

        # Draw dots: thumb (yellow), index (green), middle (purple)
        h, w = img.shape[:2]
        pts_colors = [
            (pts['thumb_tip'], (0,255,255)),     # yellow
            (pts['index_tip'], (0,255,0)),       # green
            (pts['middle_tip'], (255,0,255)),    # purple
        ]
        for p, color in pts_colors:
            cx, cy = int(p.x * w), int(p.y * h)
            cv2.circle(img, (cx, cy), 6, color, -1)

        return img

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        cap.set(cv2.CAP_PROP_FPS, self.request_fps)

        print("Hand Mouse Controller Started!")
        print("Gestures:")
        print("- Move: index finger up (pointer follows index tip)")
        print("- Left click: thumb tip CLOSE to index tip")
        print("- Right click: thumb tip CLOSE to middle tip")
        print("- Drag: maintain left click gesture while moving")
        print("- Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                lm_list = hand_landmarks.landmark

                try:
                    pts = self.get_points(lm_list)
                    gestures = self.detect_gestures(pts)
                    self.handle_actions(gestures)
                    frame = self.draw_overlay(frame, hand_landmarks, gestures, pts)
                except (TypeError, ValueError) as e:
                    print(f"[Landmark error] {e}")

            cv2.imshow('Hand Mouse Controller', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        HandMouseController().run()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"An error occurred: {e}")
