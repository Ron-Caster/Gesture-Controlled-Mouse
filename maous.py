import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

class HandMouseController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.screen_width, self.screen_height = pyautogui.size()
        self.cam_width, self.cam_height = 640, 480
        self.request_fps = 60

        self.smoothing = 10
        self.prev_x, self.prev_y = 0, 0

        self.left_cooldown = 0.2
        self.right_cooldown = 0.2
        self.last_left_time = 0.0
        self.last_right_time = 0.0

        self.touch_frames_required = 2
        self.release_frames_required = 2
        self.left_touch_frames = 0
        self.right_touch_frames = 0
        self.left_release_frames = 0
        self.right_release_frames = 0

        self.left_click_active = False
        self.right_click_active = False

        self.min_touch_threshold = 0.035
        self.require_index_up_for_clicks = True

        self.last_dm = 0.0
        self.last_dr = 0.0
        self.last_dyn_thresh = 0.0
        self.last_index_up = False

        pyautogui.FAILSAFE = False

        # Landmark indices for readability
        self.IDX = {
            'WRIST': 0,
            'THUMB_IP': 3,
            'THUMB_TIP': 4,
            'INDEX_MCP': 5,
            'INDEX_PIP': 6,
            'INDEX_TIP': 8,
            'MIDDLE_PIP': 10,
            'MIDDLE_TIP': 12,
            'RING_PIP': 14,
            'RING_TIP': 16,
            'PINKY_TIP': 20,
        }

    def get_finger_positions(self, lm_list):
        """
        lm_list must be the list of landmarks: hand_landmarks.landmark
        Returns a dict of individual NormalizedLandmark points.
        """
        if not hasattr(lm_list, '__getitem__'):
            raise TypeError("lm_list is not indexable. Did you pass hand_landmarks instead of hand_landmarks.landmark?")
        # Basic sanity check: MediaPipe returns 21 landmarks
        if len(lm_list) < 21:
            raise ValueError(f"Expected 21 landmarks, got {len(lm_list)}")

        return {
            'wrist': lm_list[self.IDX['WRIST']],
            'thumb_tip': lm_list[self.IDX['THUMB_TIP']],
            'thumb_ip': lm_list[self.IDX['THUMB_IP']],
            'index_mcp': lm_list[self.IDX['INDEX_MCP']],
            'index_pip': lm_list[self.IDX['INDEX_PIP']],
            'index_tip': lm_list[self.IDX['INDEX_TIP']],
            'middle_pip': lm_list[self.IDX['MIDDLE_PIP']],
            'middle_tip': lm_list[self.IDX['MIDDLE_TIP']],
            'ring_pip': lm_list[self.IDX['RING_PIP']],
            'ring_tip': lm_list[self.IDX['RING_TIP']],
            'pinky_tip': lm_list[self.IDX['PINKY_TIP']],
        }

    def is_finger_up(self, tip, pip):
        # tip and pip must be individual landmarks
        return tip.y < pip.y

    def calculate_distance(self, p1, p2):
        # p1 and p2 must be individual landmarks (with .x, .y)
        if not (hasattr(p1, 'x') and hasattr(p1, 'y') and hasattr(p2, 'x') and hasattr(p2, 'y')):
            raise TypeError("calculate_distance received non-landmark objects")
        return float(np.hypot(p1.x - p2.x, p1.y - p2.y))

    def detect_gestures(self, fingers):
        ref_len = self.calculate_distance(fingers['wrist'], fingers['index_mcp'])
        dyn_thresh = max(self.min_touch_threshold, ref_len * 0.9)
        self.last_dyn_thresh = float(dyn_thresh)

        index_up = self.is_finger_up(fingers['index_tip'], fingers['index_pip'])
        self.last_index_up = bool(index_up)

        d_tm_tip = self.calculate_distance(fingers['thumb_tip'], fingers['middle_tip'])
        d_tm_pip = self.calculate_distance(fingers['thumb_tip'], fingers['middle_pip'])
        d_tr_tip = self.calculate_distance(fingers['thumb_tip'], fingers['ring_tip'])
        d_tr_pip = self.calculate_distance(fingers['thumb_tip'], fingers['ring_pip'])

        self.last_dm = float(min(d_tm_tip, d_tm_pip))
        self.last_dr = float(min(d_tr_tip, d_tr_pip))

        left_touch_now = (d_tm_tip < dyn_thresh) or (d_tm_pip < dyn_thresh)
        right_touch_now = (d_tr_tip < dyn_thresh) or (d_tr_pip < dyn_thresh)

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
            'pointer_pos': fingers['index_tip']
        }

    def smooth_movement(self, x, y):
        smooth_x = self.prev_x + (x - self.prev_x) / self.smoothing
        smooth_y = self.prev_y + (y - self.prev_y) / self.smoothing
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return int(smooth_x), int(smooth_y)

    def handle_mouse_actions(self, gestures):
        now = time.time()

        if gestures['pointing']:
            pos = gestures['pointer_pos']
            screen_x = int(pos.x * self.screen_width)
            screen_y = int(pos.y * self.screen_height)
            smooth_x, smooth_y = self.smooth_movement(screen_x, screen_y)
            pyautogui.moveTo(smooth_x, smooth_y)

        clicks_allowed = (gestures['pointing'] or not self.require_index_up_for_clicks)

        if clicks_allowed and gestures['left_click'] and not self.left_click_active:
            if (now - self.last_left_time) > self.left_cooldown:
                pyautogui.click(button='left')
                self.left_click_active = True
                self.last_left_time = now
                print("Left Click")
        elif not gestures['left_click']:
            self.left_click_active = False

        if clicks_allowed and gestures['right_click'] and not self.right_click_active:
            if (now - self.last_right_time) > self.right_cooldown:
                pyautogui.click(button='right')
                self.right_click_active = True
                self.last_right_time = now
                print("Right Click")
        elif not gestures['right_click']:
            self.right_click_active = False

    def draw_landmarks_and_gestures(self, image, hand_landmarks, gestures, fingers):
        self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        cv2.putText(image, f"Pointing: {gestures['pointing']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if gestures['pointing'] else (0, 0, 255), 2)
        cv2.putText(image, f"Left Click: {gestures['left_click']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if gestures['left_click'] else (0, 0, 255), 2)
        cv2.putText(image, f"Right Click: {gestures['right_click']}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if gestures['right_click'] else (0, 0, 255), 2)

        cv2.putText(image, f"dTM(min): {self.last_dm:.3f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(image, f"dTR(min): {self.last_dr:.3f}", (10, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(image, f"dyn_thresh: {self.last_dyn_thresh:.3f}", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

        h, w = image.shape[:2]
        for name, color in [
            ('thumb_tip', (0, 255, 255)),
            ('middle_tip', (255, 0, 255)),
            ('ring_tip', (255, 255, 0)),
            ('index_tip', (0, 255, 0)),
        ]:
            p = fingers[name]
            cx, cy = int(p.x * w), int(p.y * h)
            cv2.circle(image, (cx, cy), 6, color, -1)

        return image

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        cap.set(cv2.CAP_PROP_FPS, self.request_fps)

        print("Hand Mouse Controller Started!")
        print("Gestures:")
        print("- Point with index finger to move mouse")
        print("- Touch thumb + middle finger for left click")
        print("- Touch thumb + ring finger for right click")
        print("- Maintain left click gesture while moving to drag")
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
                # We handle only the first hand due to max_num_hands=1
                hand_landmarks = results.multi_hand_landmarks[0]

                try:
                    fingers = self.get_finger_positions(hand_landmarks.landmark)  # IMPORTANT: .landmark here
                    gestures = self.detect_gestures(fingers)
                    self.handle_mouse_actions(gestures)
                    frame = self.draw_landmarks_and_gestures(frame, hand_landmarks, gestures, fingers)
                except (TypeError, ValueError) as e:
                    # Print a concise diagnostic and continue
                    print(f"[Landmark error] {e}")

            cv2.imshow('Hand Mouse Controller', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        controller = HandMouseController()
        controller.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
