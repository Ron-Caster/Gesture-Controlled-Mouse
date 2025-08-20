import cv2
import mediapipe as mp
import pyautogui
import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
import time

class HandGestureMouseControl:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
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
        self.root.geometry("340x200")

        ttk.Label(self.root, text="Adjust Circle Radius (1-100):").pack(pady=10)

        self.radius_slider = ttk.Scale(
            self.root,
            from_=1,
            to=100,
            orient="horizontal",
            command=self.update_radius,
        )
        self.radius_slider.set(self.radius)
        self.radius_slider.pack(fill="x", padx=10)

        # Pointer landmark selection (which hand node controls the cursor)
        self.landmark_names = {
            0: "WRIST",
            1: "THUMB_CMC",
            2: "THUMB_MCP",
            3: "THUMB_IP",
            4: "THUMB_TIP",
            5: "INDEX_MCP",
            6: "INDEX_PIP",
            7: "INDEX_DIP",
            8: "INDEX_TIP",
            9: "MIDDLE_MCP",
            10: "MIDDLE_PIP",
            11: "MIDDLE_DIP",
            12: "MIDDLE_TIP",
            13: "RING_MCP",
            14: "RING_PIP",
            15: "RING_DIP",
            16: "RING_TIP",
            17: "PINKY_MCP",
            18: "PINKY_PIP",
            19: "PINKY_DIP",
            20: "PINKY_TIP",
        }
        self.pointer_landmark_id = 4  # default to THUMB_TIP
        values = [f"{i}: {name}" for i, name in self.landmark_names.items()]
        ttk.Label(self.root, text="Pointer node:").pack(pady=(8, 0))
        self.pointer_choice = tk.StringVar(value=f"4: {self.landmark_names[4]}")
        self.pointer_combo = ttk.Combobox(
            self.root,
            textvariable=self.pointer_choice,
            values=values,
            state="readonly",
        )
        self.pointer_combo.pack(fill="x", padx=10)
        self.pointer_combo.bind("<<ComboboxSelected>>", self.update_pointer_landmark)

        # Thread coordination and cleanup helpers
        self.stop_event = threading.Event()
        self.video_thread = None
        self.cap = None
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Reduce OpenCV CPU usage (if supported)
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

        # Start video in separate thread
        self.video_thread = threading.Thread(target=self.run_video, daemon=True)
        self.video_thread.start()
        self.root.mainloop()

    def update_radius(self, val):
        self.radius = int(float(val))

    def update_pointer_landmark(self, event=None):
        # Parse selected combobox value and update the pointer landmark id
        try:
            sel = self.pointer_choice.get()
            lm_id = int(sel.split(":")[0].strip())
            if 0 <= lm_id <= 20:
                self.pointer_landmark_id = lm_id
        except Exception:
            pass

    def on_close(self):
        # Signal the video loop to stop and perform cleanup.
        if not self.stop_event.is_set():
            self.stop_event.set()
        # Try to release camera and close windows immediately to unblock reads.
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        # Wait briefly for the thread to finish.
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.join(timeout=2.0)
        # Final cleanup and close GUI.
        self.cleanup()
        try:
            self.root.destroy()
        except Exception:
            try:
                self.root.quit()
            except Exception:
                pass

    def cleanup(self):
        # Release any pressed mouse buttons
        try:
            if self.left_down:
                pyautogui.mouseUp(button='left')
                self.left_down = False
        except Exception:
            pass
        try:
            if self.right_down:
                pyautogui.mouseUp(button='right')
                self.right_down = False
        except Exception:
            pass
        # Release camera
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        except Exception:
            pass
        # Close MediaPipe resources
        try:
            if self.hands is not None:
                self.hands.close()
                self.hands = None
        except Exception:
            pass
        # Close OpenCV windows
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def run_video(self):
        # Use DirectShow backend on Windows for stability and set modest resolution/FPS.
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass

        try:
            last_tick = time.perf_counter()
            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    # Backoff a bit if capture fails to avoid a busy loop
                    time.sleep(0.02)
                    continue

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

                    # Mouse movement using selected pointer landmark
                    try:
                        lm_ptr = hand_landmarks.landmark[self.pointer_landmark_id]
                        px, py = int(lm_ptr.x * w), int(lm_ptr.y * h)
                        mouse_x = int(np.interp(px, (0, w), (0, self.screen_width)))
                        mouse_y = int(np.interp(py, (0, h), (0, self.screen_height)))
                        pyautogui.moveTo(mouse_x, mouse_y)
                        # Highlight the selected pointer node
                        cv2.circle(frame, (px, py), max(8, self.radius), (0, 0, 255), 2)
                        cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)
                    except Exception:
                        pass

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

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    self.stop_event.set()
                    break

                # Throttle to ~60 FPS max to reduce CPU
                now = time.perf_counter()
                dt = now - last_tick
                if dt < (1.0 / 60.0):
                    time.sleep((1.0 / 60.0) - dt)
                last_tick = now
        except Exception as e:
            print(f"Video loop error: {e}")
        finally:
            # Ensure resources are released
            try:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
            except Exception:
                pass
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            try:
                if self.hands is not None:
                    self.hands.close()
                    self.hands = None
            except Exception:
                pass
            # Make sure mouse buttons are not left pressed
            try:
                if self.left_down:
                    pyautogui.mouseUp(button='left')
                    self.left_down = False
            except Exception:
                pass
            try:
                if self.right_down:
                    pyautogui.mouseUp(button='right')
                    self.right_down = False
            except Exception:
                pass
        # Close GUI after loop ends if not already closing
        try:
            self.root.after(0, self.on_close)
        except Exception:
            pass

if __name__ == "__main__":
    HandGestureMouseControl()
