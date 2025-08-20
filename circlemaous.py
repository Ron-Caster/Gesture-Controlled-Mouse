import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
import time
from collections import deque
from queue import Queue, Empty

# Optional lower-level mouse backend
USE_PYNPUT = False
try:
    from pynput.mouse import Controller as PynputMouse, Button as PynputButton
    USE_PYNPUT = True
except Exception:
    import pyautogui
    pyautogui.FAILSAFE = False  # avoid exceptions on corners


class HandGestureMouseControl:
    def __init__(self):
        # --- MediaPipe setup ---
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # --- Mouse backend ---
        if USE_PYNPUT:
            self.mouse = PynputMouse()
            self.btn_left = PynputButton.left
            self.btn_right = PynputButton.right
            # screen size fallback via Tk
            root_tmp = tk.Tk(); root_tmp.withdraw()
            self.screen_width = root_tmp.winfo_screenwidth()
            self.screen_height = root_tmp.winfo_screenheight()
            root_tmp.destroy()
        else:
            # Fallback: pyautogui
            self.screen_width, self.screen_height = pyautogui.size()

        # --- State ---
        self.radius = 20
        self.left_down = False
        self.right_down = False

        # Debounce counters
        self.left_touch_frames = 0
        self.left_release_frames = 0
        self.right_touch_frames = 0
        self.right_release_frames = 0

        # Thresholds (tunable from UI)
        self.touch_threshold_frames = 2
        self.release_threshold_frames = 2

        # EWMA smoothing
        self.alpha = 0.7
        self.smooth_x = None
        self.smooth_y = None

        # Frame processing options
        self.proc_width = 320
        self.proc_height = 240
        self.render_enabled = True
        self.detection_stride = 1  # process every frame; set >1 to skip
        self._frame_index = 0

        # Interpolation history when skipping frames
        self.last_px_py = None
        self.prev_px_py = None

        # Pointer landmark selector
        self.landmark_names = {
            0: "WRIST", 1: "THUMB_CMC", 2: "THUMB_MCP", 3: "THUMB_IP", 4: "THUMB_TIP",
            5: "INDEX_MCP", 6: "INDEX_PIP", 7: "INDEX_DIP", 8: "INDEX_TIP",
            9: "MIDDLE_MCP", 10: "MIDDLE_PIP", 11: "MIDDLE_DIP", 12: "MIDDLE_TIP",
            13: "RING_MCP", 14: "RING_PIP", 15: "RING_DIP", 16: "RING_TIP",
            17: "PINKY_MCP", 18: "PINKY_PIP", 19: "PINKY_DIP", 20: "PINKY_TIP",
        }
        self.pointer_landmark_id = 4  # THUMB_TIP

        # Threading / Queues
        self.stop_event = threading.Event()
        self.cap = None
        self.frame_q: Queue = Queue(maxsize=2)  # small buffer to keep latency low

        # Tkinter GUI
        self.root = tk.Tk()
        self.root.title("Hand Gesture Mouse Control (Low-Latency)")
        self.root.geometry("420x320")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # UI: radius
        ttk.Label(self.root, text="Circle Radius (1-100):").pack(pady=(10, 0))
        self.radius_slider = ttk.Scale(self.root, from_=1, to=100, orient="horizontal",
                                       command=self.update_radius)
        self.radius_slider.set(self.radius); self.radius_slider.pack(fill="x", padx=10)

        # UI: pointer node
        ttk.Label(self.root, text="Pointer node:").pack(pady=(8, 0))
        values = [f"{i}: {name}" for i, name in self.landmark_names.items()]
        self.pointer_choice = tk.StringVar(value=f"4: {self.landmark_names[4]}")
        self.pointer_combo = ttk.Combobox(self.root, textvariable=self.pointer_choice,
                                          values=values, state="readonly")
        self.pointer_combo.pack(fill="x", padx=10)
        self.pointer_combo.bind("<<ComboboxSelected>>", self.update_pointer_landmark)

        # UI: smoothing alpha
        ttk.Label(self.root, text="Smoothing α (0.0–1.0):").pack(pady=(8, 0))
        self.alpha_slider = ttk.Scale(self.root, from_=0.0, to=1.0, orient="horizontal",
                                      command=self.update_alpha)
        self.alpha_slider.set(self.alpha); self.alpha_slider.pack(fill="x", padx=10)

        # UI: debounce thresholds
        ttk.Label(self.root, text="Touch frames (click down):").pack(pady=(8, 0))
        self.touch_slider = ttk.Scale(self.root, from_=1, to=5, orient="horizontal",
                                      command=self.update_touch_threshold)
        self.touch_slider.set(self.touch_threshold_frames); self.touch_slider.pack(fill="x", padx=10)

        ttk.Label(self.root, text="Release frames (click up):").pack(pady=(8, 0))
        self.release_slider = ttk.Scale(self.root, from_=1, to=5, orient="horizontal",
                                        command=self.update_release_threshold)
        self.release_slider.set(self.release_threshold_frames); self.release_slider.pack(fill="x", padx=10)

        # UI: detection stride
        ttk.Label(self.root, text="Detection stride (1=every frame):").pack(pady=(8, 0))
        self.stride_slider = ttk.Scale(self.root, from_=1, to=4, orient="horizontal",
                                       command=self.update_stride)
        self.stride_slider.set(self.detection_stride); self.stride_slider.pack(fill="x", padx=10)

        # UI: render toggle
        self.render_var = tk.BooleanVar(value=self.render_enabled)
        self.render_chk = ttk.Checkbutton(self.root, text="Render preview window (disable for lowest latency)",
                                          variable=self.render_var, command=self.toggle_render)
        self.render_chk.pack(pady=(8, 0))

        # UI: backend label
        backend = "pynput (low-level)" if USE_PYNPUT else "pyautogui (fallback)"
        ttk.Label(self.root, text=f"Mouse backend: {backend}").pack(pady=(8, 0))

        # Start threads
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.process_thread = threading.Thread(target=self.process_loop, daemon=True)
        self.capture_thread.start()
        self.process_thread.start()

        # Main loop
        self.root.mainloop()

    # --- UI handlers ---
    def update_radius(self, val): self.radius = int(float(val))
    def update_alpha(self, val): self.alpha = float(val)
    def update_touch_threshold(self, val): self.touch_threshold_frames = int(float(val))
    def update_release_threshold(self, val): self.release_threshold_frames = int(float(val))
    def update_stride(self, val): self.detection_stride = max(1, int(float(val)))
    def toggle_render(self): self.render_enabled = self.render_var.get()

    def update_pointer_landmark(self, event=None):
        try:
            sel = self.pointer_choice.get()
            lm_id = int(sel.split(":")[0].strip())
            if 0 <= lm_id <= 20:
                self.pointer_landmark_id = lm_id
        except Exception:
            pass

    # --- Shutdown / cleanup ---
    def on_close(self):
        if not self.stop_event.is_set():
            self.stop_event.set()
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            self.hands.close()
        except Exception:
            pass

        # Ensure mouse buttons up
        self.release_mouse_buttons()

        try:
            self.root.destroy()
        except Exception:
            try:
                self.root.quit()
            except Exception:
                pass

    def release_mouse_buttons(self):
        try:
            if self.left_down:
                if USE_PYNPUT:
                    self.mouse.release(self.btn_left)
                else:
                    import pyautogui
                    pyautogui.mouseUp(button='left')
                self.left_down = False
        except Exception:
            pass
        try:
            if self.right_down:
                if USE_PYNPUT:
                    self.mouse.release(self.btn_right)
                else:
                    import pyautogui
                    pyautogui.mouseUp(button='right')
                self.right_down = False
        except Exception:
            pass

    # --- Threads ---
    def capture_loop(self):
        # Prefer DirectShow on Windows
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            # Mirror
            frame = cv2.flip(frame, 1)
            # Keep newest frame only (drop older if queue full)
            try:
                if self.frame_q.full():
                    try:
                        self.frame_q.get_nowait()
                    except Empty:
                        pass
                self.frame_q.put_nowait(frame)
            except Exception:
                pass

        # Clean up
        try:
            self.cap.release()
        except Exception:
            pass

    def process_loop(self):
        last_tick = time.perf_counter()
        window_name = "Finger Circles and Mouse Control"
        while not self.stop_event.is_set():
            try:
                frame = self.frame_q.get(timeout=0.05)
            except Empty:
                continue

            self._frame_index += 1
            h0, w0 = frame.shape[:2]

            # Downscale for processing
            proc = cv2.resize(frame, (self.proc_width, self.proc_height), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

            run_detection = (self._frame_index % self.detection_stride == 0)
            if run_detection:
                results = self.hands.process(rgb)
            else:
                results = None

            # Use last landmarks if skipping
            landmarks = None
            if results and results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
            elif self.last_px_py is not None:
                # No new detection; we will interpolate pointer using last / prev
                pass

            disp = frame.copy()
            px, py = None, None
            tip_positions = {}

            if landmarks is not None:
                # Draw and compute in processed coords; map to original frame size
                # Fingertips: 4 (thumb), 8 (index), 12 (middle)
                for lm_id in (4, 8, 12):
                    lm = landmarks.landmark[lm_id]
                    cx_p, cy_p = int(lm.x * self.proc_width), int(lm.y * self.proc_height)
                    # Map to original frame for drawings
                    cx = int((cx_p / self.proc_width) * w0)
                    cy = int((cy_p / self.proc_height) * h0)
                    tip_positions[lm_id] = (cx, cy)
                    if self.render_enabled:
                        cv2.circle(disp, (cx, cy), self.radius, (0, 255, 255) if lm_id == 4 else
                                   ((0, 255, 0) if lm_id == 8 else (255, 0, 255)), 2)

                # Pointer node
                try:
                    lm_ptr = landmarks.landmark[self.pointer_landmark_id]
                    px_p, py_p = int(lm_ptr.x * self.proc_width), int(lm_ptr.y * self.proc_height)
                    # Map to original
                    px = int((px_p / self.proc_width) * w0)
                    py = int((py_p / self.proc_height) * h0)
                    if self.render_enabled:
                        cv2.circle(disp, (px, py), max(8, self.radius), (0, 0, 255), 2)
                        cv2.circle(disp, (px, py), 3, (0, 0, 255), -1)
                    # Maintain interpolation history
                    self.prev_px_py = self.last_px_py
                    self.last_px_py = (px, py)
                except Exception:
                    pass
            else:
                # Interpolate pointer between prev and last when skipping frames
                if self.prev_px_py and self.last_px_py:
                    t = 1.0 / max(1, self.detection_stride)  # simple step
                    px = int((1 - t) * self.prev_px_py[0] + t * self.last_px_py[0])
                    py = int((1 - t) * self.prev_px_py[1] + t * self.last_px_py[1])

            # Mouse movement with EWMA smoothing
            if px is not None and py is not None:
                mouse_x = int(np.interp(px, (0, w0), (0, self.screen_width)))
                mouse_y = int(np.interp(py, (0, h0), (0, self.screen_height)))

                if self.smooth_x is None:
                    self.smooth_x, self.smooth_y = mouse_x, mouse_y
                else:
                    a = self.alpha
                    self.smooth_x = int(a * mouse_x + (1 - a) * self.smooth_x)
                    self.smooth_y = int(a * mouse_y + (1 - a) * self.smooth_y)

                self._move_mouse(self.smooth_x, self.smooth_y)

            # Click logic (debounced) using overlap in original-sized coords
            if len(tip_positions) >= 3:
                # Distances
                ti = np.array(tip_positions.get(4, (0, 0)))
                ii = np.array(tip_positions.get(8, (10**9, 10**9)))
                mi = np.array(tip_positions.get(12, (10**9, 10**9)))
                dist_thumb_index = float(np.linalg.norm(ti - ii))
                dist_thumb_middle = float(np.linalg.norm(ti - mi))
                overlap_thr = 2 * self.radius

                # Left click
                if dist_thumb_index < overlap_thr:
                    self.left_touch_frames += 1; self.left_release_frames = 0
                    if self.left_touch_frames >= self.touch_threshold_frames and not self.left_down:
                        self._mouse_down('left')
                        self.left_down = True
                else:
                    self.left_release_frames += 1
                    if self.left_release_frames >= self.release_threshold_frames and self.left_down:
                        self._mouse_up('left')
                        self.left_down = False
                        self.left_touch_frames = 0

                # Right click
                if dist_thumb_middle < overlap_thr:
                    self.right_touch_frames += 1; self.right_release_frames = 0
                    if self.right_touch_frames >= self.touch_threshold_frames and not self.right_down:
                        self._mouse_down('right')
                        self.right_down = True
                else:
                    self.right_release_frames += 1
                    if self.right_release_frames >= self.release_threshold_frames and self.right_down:
                        self._mouse_up('right')
                        self.right_down = False
                        self.right_touch_frames = 0

            # Render (optional)
            if self.render_enabled:
                try:
                    cv2.imshow(window_name, disp)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        self.stop_event.set()
                        break
                except Exception:
                    pass

            # Throttle to ~60 FPS max (processing thread)
            now = time.perf_counter()
            dt = now - last_tick
            if dt < (1.0 / 60.0):
                time.sleep((1.0 / 60.0) - dt)
            last_tick = now

        # Cleanup
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.release_mouse_buttons()
        try:
            self.hands.close()
        except Exception:
            pass

        # Close app after loop ends
        try:
            self.root.after(0, self.on_close)
        except Exception:
            pass

    # --- Mouse helpers ---
    def _move_mouse(self, x, y):
        try:
            if USE_PYNPUT:
                self.mouse.position = (x, y)
            else:
                import pyautogui
                pyautogui.moveTo(x, y, _pause=False)
        except Exception:
            pass

    def _mouse_down(self, which):
        try:
            if USE_PYNPUT:
                self.mouse.press(self.btn_left if which == 'left' else self.btn_right)
            else:
                import pyautogui
                pyautogui.mouseDown(button=which)
        except Exception:
            pass

    def _mouse_up(self, which):
        try:
            if USE_PYNPUT:
                self.mouse.release(self.btn_left if which == 'left' else self.btn_right)
            else:
                import pyautogui
                pyautogui.mouseUp(button=which)
        except Exception:
            pass


if __name__ == "__main__":
    HandGestureMouseControl()
