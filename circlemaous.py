import cv2
import mediapipe as mp
import pyautogui
import tkinter as tk
from tkinter import ttk
import threading
import numpy as np

class HandGestureMouseControl:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Initial radius
        self.radius = 20  # Starting pixel radius for circles and overlap detection

        # Mouse control states
        self.left_down = False
        self.right_down = False

        # Debounce counters
        self.left_touch_frames = 0
        self.left_release_frames = 0
        self.right_touch_frames = 0
        self.right_release_frames = 0

        # Stability thresholds
        self.touch_threshold_frames = 2
        self.release_threshold_frames = 2

        # Screen size for mouse mapping
        self.screen_width, self.screen_height = pyautogui.size()

        # Tkinter GUI
        self.root = tk.Tk()
        self.root.title("Circle Radius Adjuster")
        self.root.geometry("300x100")

        ttk.Label(self.root, text="Adjust Circle Radius (1-100):").pack(pady=10)

        self.radius_slider = ttk.Scale(self.root, from_=1, to=100, orient='horizontal',
                                       command=self.update_radius)
        self.radius_slider.set(self.radius)
        self.radius_slider.pack(fill='x', padx=10)

        # Start video in separate thread
        threading.Thread(target=self.run_video, daemon=True).start()
        self.root.mainloop()

    def update_radius(self, val):
        self.radius = int(float(val))

    def run_video(self):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            h, w = frame.shape[:2]

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Fingertip landmarks: thumb (4), index (8), middle (12)
                fingertips = [4, 8, 12]
                colors = [(0, 255, 255), (0, 255, 0), (255, 0, 255)]  # Yellow, Green, Purple

                tip_positions = {}
                for idx, lm_id in enumerate(fingertips):
                    lm = hand_landmarks.landmark[lm_id]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    tip_positions[lm_id] = (cx, cy)
                    cv2.circle(frame, (cx, cy), self.radius, colors[idx], 2)

                # Mouse movement using thumb tip
                thumb_pos = tip_positions[4]
                mouse_x = int(np.interp(thumb_pos[0], (0, w), (0, self.screen_width)))
                mouse_y = int(np.interp(thumb_pos[1], (0, h), (0, self.screen_height)))
                pyautogui.moveTo(mouse_x, mouse_y)

                # Calculate distances for overlap detection
                dist_thumb_index = np.linalg.norm(np.array(tip_positions[4]) - np.array(tip_positions[8]))
                dist_thumb_middle = np.linalg.norm(np.array(tip_positions[4]) - np.array(tip_positions[12]))

                # Overlap threshold: 2 * radius (assuming equal radii)
                overlap_thr = 2 * self.radius

                # Left click detection (thumb-index overlap)
                if dist_thumb_index < overlap_thr:
                    self.left_touch_frames += 1
                    self.left_release_frames = 0
                    if self.left_touch_frames >= self.touch_threshold_frames and not self.left_down:
                        pyautogui.mouseDown(button='left')
                        self.left_down = True
                        print("Left Down")
                else:
                    self.left_release_frames += 1
                    if self.left_release_frames >= self.release_threshold_frames and self.left_down:
                        pyautogui.mouseUp(button='left')
                        self.left_down = False
                        self.left_touch_frames = 0
                        print("Left Up")

                # Right click detection (thumb-middle overlap)
                if dist_thumb_middle < overlap_thr:
                    self.right_touch_frames += 1
                    self.right_release_frames = 0
                    if self.right_touch_frames >= self.touch_threshold_frames and not self.right_down:
                        pyautogui.mouseDown(button='right')
                        self.right_down = True
                        print("Right Down")
                else:
                    self.right_release_frames += 1
                    if self.right_release_frames >= self.release_threshold_frames and self.right_down:
                        pyautogui.mouseUp(button='right')
                        self.right_down = False
                        self.right_touch_frames = 0
                        print("Right Up")

            cv2.imshow("Finger Circles and Mouse Control", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.root.quit()  # Close GUI when video loop ends

if __name__ == "__main__":
    HandGestureMouseControl()
