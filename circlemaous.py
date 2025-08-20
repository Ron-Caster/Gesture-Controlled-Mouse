"""
Full single-file program: Hand gesture virtual mouse with low-latency options.

Features included:
- Downscaled processing (320x240) with ROI option.
- EWMA smoothing, optional Kalman filter (constant-velocity).
- Optional velocity-based prediction (look-ahead ms).
- Optional GPU path for resize/color using OpenCV CUDA (safe fallback).
- Debounced click detection (thumb-index / thumb-middle).
- Capture & processing in separate threads (small queue).
- GUI (Tkinter) with checkboxes for:
    * Kalman filter (OFF default)
    * Velocity prediction (OFF default)
    * ROI cropping (OFF default)
    * GPU path (OFF default)
    * Release control when no hand detected (idle timeout) (OFF default)
    * Enable hotkey (F8) to toggle virtual mouse (OFF default)
- Hotkey (F8) toggles virtual mouse ON/OFF when "Enable hotkey" is checked.
- Release-on-idle: when enabled and no hand seen for idle timeout, stops sending mouse updates
  and releases mouse buttons so physical mouse regains control.
- Safe cleanup on exit.

Requirements:
- Python packages: opencv-python, mediapipe, numpy, tkinter (standard), pynput (optional), pyautogui (fallback)
- On Windows the code prefers DirectShow backend for capture.

Copy & paste to a file (e.g., hand_mouse.py) and run.
"""

import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
import threading
import numpy as np
import time
from queue import Queue, Empty

# Optional low-level mouse backend
USE_PYNPUT_MOUSE = False
try:
    from pynput.mouse import Controller as PynputMouse, Button as PynputButton
    USE_PYNPUT_MOUSE = True
except Exception:
    import pyautogui
    pyautogui.FAILSAFE = False  # avoid corner exceptions

# Optional keyboard/hotkey support via pynput (preferred) or keyboard module fallback
HOTKEY_PYNPUT_AVAILABLE = False
HOTKEY_KEY = "f8"
try:
    from pynput import keyboard as pynput_keyboard
    HOTKEY_PYNPUT_AVAILABLE = True
except Exception:
    try:
        import keyboard as kb_fallback  # pip install keyboard (may require admin on some OS)
        HOTKEY_PYNPUT_AVAILABLE = False
    except Exception:
        kb_fallback = None
        HOTKEY_PYNPUT_AVAILABLE = False

# ---------------- Kalman (2D constant-velocity) ----------------
class Kalman2D:
    """
    State: [x, y, vx, vy]^T
    Very small, numerically-stable implementation for smoothing/prediction.
    """
    def __init__(self, process_var=5e-3, meas_var=6.0):
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 1e3
        self.Q_base = np.array([[1/4, 0, 1/2, 0],
                                [0, 1/4, 0, 1/2],
                                [1/2, 0, 1, 0],
                                [0, 1/2, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        self.R = np.eye(2, dtype=np.float32) * meas_var
        self.process_var = process_var
        self.last_t = None

    def predict(self, dt):
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        dt2 = dt * dt
        Q = self.Q_base.copy()
        Q[0, 0] *= dt2
        Q[1, 1] *= dt2
        Q[0, 2] *= dt
        Q[1, 3] *= dt
        Q[2, 0] *= dt
        Q[3, 1] *= dt
        Q *= self.process_var

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

    def correct(self, meas_x, meas_y, dt):
        if dt <= 0:
            dt = 1/60.0
        self.predict(dt)
        z = np.array([[meas_x], [meas_y]], dtype=np.float32)
        self.update(z)
        return float(self.x[0, 0]), float(self.x[1, 0])

    def get_predicted_pos(self, lookahead_s):
        px = float(self.x[0, 0] + self.x[2, 0] * lookahead_s)
        py = float(self.x[1, 0] + self.x[3, 0] * lookahead_s)
        return px, py

# ---------------- Main class ----------------
class HandGestureMouseControl:
    def __init__(self):
        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Mouse backend
        if USE_PYNPUT_MOUSE:
            self.mouse = PynputMouse()
            self.btn_left = PynputButton.left
            self.btn_right = PynputButton.right
            # screen size
            root_tmp = tk.Tk(); root_tmp.withdraw()
            self.screen_width = root_tmp.winfo_screenwidth()
            self.screen_height = root_tmp.winfo_screenheight()
            root_tmp.destroy()
        else:
            import pyautogui
            self.screen_width, self.screen_height = pyautogui.size()

        # --- state / defaults ---
        self.radius = 20
        self.left_down = False
        self.right_down = False

        # debounce counters
        self.left_touch_frames = 0
        self.left_release_frames = 0
        self.right_touch_frames = 0
        self.right_release_frames = 0

        # thresholds
        self.touch_threshold_frames = 2
        self.release_threshold_frames = 2

        # smoothing
        self.alpha = 0.7
        self.smooth_x = None
        self.smooth_y = None

        # Kalman
        self.kalman_enabled = False
        self.kf = Kalman2D(process_var=5e-3, meas_var=6.0)

        # prediction
        self.prediction_enabled = False
        self.prediction_ms = 20

        # downscale processing
        self.proc_width = 320
        self.proc_height = 240
        self.render_enabled = True
        self.detection_stride = 1
        self._frame_index = 0

        # interpolation
        self.last_px_py = None
        self.prev_px_py = None

        # ROI
        self.roi_enabled = False
        self.roi_margin = 0.25
        self.last_landmarks = None

        # GPU path
        self.gpu_enabled = False
        self.cuda_ok = False
        self._init_cuda()

        # pointer landmark
        self.landmark_names = {
            0: "WRIST", 1: "THUMB_CMC", 2: "THUMB_MCP", 3: "THUMB_IP", 4: "THUMB_TIP",
            5: "INDEX_MCP", 6: "INDEX_PIP", 7: "INDEX_DIP", 8: "INDEX_TIP",
            9: "MIDDLE_MCP", 10: "MIDDLE_PIP", 11: "MIDDLE_DIP", 12: "MIDDLE_TIP",
            13: "RING_MCP", 14: "RING_PIP", 15: "RING_DIP", 16: "RING_TIP",
            17: "PINKY_MCP", 18: "PINKY_PIP", 19: "PINKY_DIP", 20: "PINKY_TIP",
        }
        self.pointer_landmark_id = 4  # default THUMB_TIP

        # release-on-idle option
        self.release_on_idle = False
        self.idle_timeout_ms = 1000
        self.last_detection_time = 0.0

        # virtual mouse toggle (hotkey or GUI)
        self.virtual_mouse_enabled = True
        self.hotkey_enabled = False
        self._hotkey_listener = None

        # queue + threads
        self.stop_event = threading.Event()
        self.cap = None
        self.frame_q: Queue = Queue(maxsize=2)

        # build GUI
        self._build_gui()

        # threads
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.process_thread = threading.Thread(target=self.process_loop, daemon=True)
        self.capture_thread.start()
        self.process_thread.start()

        # start hotkey listener (pynput preferred, else keyboard fallback thread)
        if HOTKEY_PYNPUT_AVAILABLE:
            self._start_pynput_hotkey_listener()
        else:
            self._start_keyboard_poll_thread()

        # main loop
        self.root.mainloop()

    # ---------- CUDA helpers ----------
    def _init_cuda(self):
        try:
            if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.cuda_ok = True
        except Exception:
            self.cuda_ok = False

    def _gpu_resize_cvt(self, frame_bgr, out_w, out_h):
        if not (self.gpu_enabled and self.cuda_ok):
            proc = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            return rgb
        try:
            gmat = cv2.cuda_GpuMat()
            gmat.upload(frame_bgr)
            gmat = cv2.cuda.resize(gmat, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            cvt = cv2.cuda.cvtColor(gmat, cv2.COLOR_BGR2RGB)
            rgb = cvt.download()
            return rgb
        except Exception:
            proc = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            return rgb

    # ---------- GUI ----------
    def _build_gui(self):
        self.root = tk.Tk()
        self.root.title("Hand Gesture Mouse Control - Options")
        self.root.geometry("560x640")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        frm_basic = ttk.LabelFrame(self.root, text="Basic")
        frm_basic.pack(fill="x", padx=10, pady=8)

        ttk.Label(frm_basic, text="Circle Radius (1-100):").pack(pady=(8, 0))
        self.radius_slider = ttk.Scale(frm_basic, from_=1, to=100, orient="horizontal",
                                       command=self.update_radius)
        self.radius_slider.set(self.radius); self.radius_slider.pack(fill="x", padx=10)

        ttk.Label(frm_basic, text="Pointer node:").pack(pady=(8, 0))
        values = [f"{i}: {name}" for i, name in self.landmark_names.items()]
        self.pointer_choice = tk.StringVar(value=f"4: {self.landmark_names[4]}")
        self.pointer_combo = ttk.Combobox(frm_basic, textvariable=self.pointer_choice,
                                          values=values, state="readonly")
        self.pointer_combo.pack(fill="x", padx=10)
        self.pointer_combo.bind("<<ComboboxSelected>>", self.update_pointer_landmark)

        ttk.Label(frm_basic, text="Smoothing α (0.0–1.0):").pack(pady=(8, 0))
        self.alpha_slider = ttk.Scale(frm_basic, from_=0.0, to=1.0, orient="horizontal",
                                      command=self.update_alpha)
        self.alpha_slider.set(self.alpha); self.alpha_slider.pack(fill="x", padx=10)

        ttk.Label(frm_basic, text="Touch frames (click down):").pack(pady=(8, 0))
        self.touch_slider = ttk.Scale(frm_basic, from_=1, to=6, orient="horizontal",
                                      command=self.update_touch_threshold)
        self.touch_slider.set(self.touch_threshold_frames); self.touch_slider.pack(fill="x", padx=10)

        ttk.Label(frm_basic, text="Release frames (click up):").pack(pady=(8, 0))
        self.release_slider = ttk.Scale(frm_basic, from_=1, to=6, orient="horizontal",
                                        command=self.update_release_threshold)
        self.release_slider.set(self.release_threshold_frames); self.release_slider.pack(fill="x", padx=10)

        ttk.Label(frm_basic, text="Detection stride (1=every frame):").pack(pady=(8, 0))
        self.stride_slider = ttk.Scale(frm_basic, from_=1, to=4, orient="horizontal",
                                       command=self.update_stride)
        self.stride_slider.set(self.detection_stride); self.stride_slider.pack(fill="x", padx=10)

        self.render_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm_basic, text="Render preview (disable for lowest latency)",
                        variable=self.render_var, command=self.toggle_render).pack(pady=(8, 6))

        backend = "pynput (low-level)" if USE_PYNPUT_MOUSE else "pyautogui (fallback)"
        ttk.Label(frm_basic, text=f"Mouse backend: {backend}").pack(pady=(0, 6))

        # Optional features
        frm_opt = ttk.LabelFrame(self.root, text="Optional Features (default OFF)")
        frm_opt.pack(fill="x", padx=10, pady=8)

        self.kalman_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_opt, text="Kalman filter smoothing", variable=self.kalman_var,
                        command=self.toggle_kalman).pack(anchor="w", padx=10, pady=(6, 2))

        self.pred_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_opt, text="Velocity-based prediction (look-ahead)", variable=self.pred_var,
                        command=self.toggle_prediction).pack(anchor="w", padx=10, pady=(6, 2))
        ttk.Label(frm_opt, text="Prediction (ms):").pack(anchor="w", padx=10)
        self.pred_ms_slider = ttk.Scale(frm_opt, from_=0, to=120, orient="horizontal",
                                        command=self.update_prediction_ms)
        self.pred_ms_slider.set(self.prediction_ms); self.pred_ms_slider.pack(fill="x", padx=10, pady=(0, 6))

        self.roi_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_opt, text="ROI cropping around last hand", variable=self.roi_var,
                        command=self.toggle_roi).pack(anchor="w", padx=10, pady=(6, 2))
        ttk.Label(frm_opt, text="ROI margin (0.05–0.5):").pack(anchor="w", padx=10)
        self.roi_margin_slider = ttk.Scale(frm_opt, from_=0.05, to=0.5, orient="horizontal",
                                           command=self.update_roi_margin)
        self.roi_margin_slider.set(self.roi_margin); self.roi_margin_slider.pack(fill="x", padx=10, pady=(0, 6))

        self.gpu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_opt, text="GPU path (OpenCV CUDA for resize/convert)", variable=self.gpu_var,
                        command=self.toggle_gpu).pack(anchor="w", padx=10, pady=(6, 2))
        gpu_status = "available" if self.cuda_ok else "NOT available"
        ttk.Label(frm_opt, text=f"CUDA status: {gpu_status}").pack(anchor="w", padx=10, pady=(0, 6))

        # New options: release-on-idle and hotkey
        self.idle_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_opt, text="Release control when no hand detected (idle timeout)", variable=self.idle_var,
                        command=self.toggle_release_on_idle).pack(anchor="w", padx=10, pady=(6, 2))
        ttk.Label(frm_opt, text="Idle timeout (ms):").pack(anchor="w", padx=10)
        self.idle_timeout_slider = ttk.Scale(frm_opt, from_=200, to=5000, orient="horizontal",
                                             command=self.update_idle_timeout)
        self.idle_timeout_slider.set(self.idle_timeout_ms); self.idle_timeout_slider.pack(fill="x", padx=10, pady=(0, 6))

        self.hotkey_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_opt, text=f"Enable hotkey ({HOTKEY_KEY.upper()}) to toggle virtual mouse", variable=self.hotkey_var,
                        command=self.toggle_hotkey_enabled).pack(anchor="w", padx=10, pady=(6, 2))

        # Virtual mouse status label & manual toggle button
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", padx=10, pady=8)
        ttk.Label(status_frame, text="Virtual Mouse:").pack(side="left", padx=(0, 6))
        self.vm_status_var = tk.StringVar(value="ON" if self.virtual_mouse_enabled else "OFF")
        self.vm_status_lbl = ttk.Label(status_frame, textvariable=self.vm_status_var, foreground="green")
        self.vm_status_lbl.pack(side="left")
        ttk.Button(status_frame, text="Toggle (GUI)", command=self.gui_toggle_virtual_mouse).pack(side="right")

        # Quit button
        ttk.Button(self.root, text="Quit", command=self.on_close).pack(pady=8)

    # ---------- UI handlers ----------
    def update_radius(self, val): self.radius = int(float(val))
    def update_alpha(self, val): self.alpha = float(val)
    def update_touch_threshold(self, val): self.touch_threshold_frames = int(float(val))
    def update_release_threshold(self, val): self.release_threshold_frames = int(float(val))
    def update_stride(self, val): self.detection_stride = max(1, int(float(val)))
    def toggle_render(self): self.render_enabled = self.render_var.get()
    def toggle_kalman(self): self.kalman_enabled = self.kalman_var.get()
    def toggle_prediction(self): self.prediction_enabled = self.pred_var.get()
    def update_prediction_ms(self, val): self.prediction_ms = int(float(val))
    def toggle_roi(self): self.roi_enabled = self.roi_var.get()
    def update_roi_margin(self, val): self.roi_margin = float(val)
    def toggle_gpu(self):
        # safe fallback
        self.gpu_enabled = self.gpu_var.get()
        if self.gpu_enabled and not self.cuda_ok:
            self.gpu_enabled = False
            self.gpu_var.set(False)
    def toggle_release_on_idle(self): self.release_on_idle = self.idle_var.get()
    def update_idle_timeout(self, val): self.idle_timeout_ms = int(float(val))
    def toggle_hotkey_enabled(self): self.hotkey_enabled = self.hotkey_var.get()
    def gui_toggle_virtual_mouse(self):
        self.virtual_mouse_enabled = not self.virtual_mouse_enabled
        self._update_vm_status_label()

    def update_pointer_landmark(self, event=None):
        try:
            sel = self.pointer_choice.get()
            lm_id = int(sel.split(":")[0].strip())
            if 0 <= lm_id <= 20:
                self.pointer_landmark_id = lm_id
        except Exception:
            pass

    def _update_vm_status_label(self):
        self.vm_status_var.set("ON" if self.virtual_mouse_enabled else "OFF")
        if self.virtual_mouse_enabled:
            self.vm_status_lbl.config(foreground="green")
        else:
            self.vm_status_lbl.config(foreground="red")

    # ---------- Hotkey listeners ----------
    def _on_pynput_press(self, key):
        try:
            if not self.hotkey_enabled:
                return
            # handle F8
            if key == getattr(pynput_keyboard.Key, HOTKEY_KEY):
                # toggle
                self.virtual_mouse_enabled = not self.virtual_mouse_enabled
                # ensure releasing mouse buttons when disabling
                if not self.virtual_mouse_enabled:
                    self.release_mouse_buttons()
                # update label in GUI thread
                try:
                    self.root.after(0, self._update_vm_status_label)
                except Exception:
                    pass
        except Exception:
            pass

    def _start_pynput_hotkey_listener(self):
        # Start a global key listener (runs in separate thread)
        def run_listener():
            try:
                self._hotkey_listener = pynput_keyboard.Listener(on_press=self._on_pynput_press)
                self._hotkey_listener.start()
                # keep it alive until stop_event
                while not self.stop_event.is_set():
                    time.sleep(0.1)
                try:
                    self._hotkey_listener.stop()
                except Exception:
                    pass
            except Exception:
                pass
        t = threading.Thread(target=run_listener, daemon=True)
        t.start()

    def _start_keyboard_poll_thread(self):
        # Fallback: poll keyboard module if available
        if kb_fallback is None:
            return
        def poll_loop():
            last_state = False
            while not self.stop_event.is_set():
                try:
                    pressed = kb_fallback.is_pressed(HOTKEY_KEY)
                    if pressed and not last_state and self.hotkey_enabled:
                        # toggle
                        self.virtual_mouse_enabled = not self.virtual_mouse_enabled
                        if not self.virtual_mouse_enabled:
                            self.release_mouse_buttons()
                        try:
                            self.root.after(0, self._update_vm_status_label)
                        except Exception:
                            pass
                        time.sleep(0.3)  # debounce
                    last_state = pressed
                except Exception:
                    pass
                time.sleep(0.05)
        t = threading.Thread(target=poll_loop, daemon=True)
        t.start()

    # ---------- Capture thread ----------
    def capture_loop(self):
        # Prefer DirectShow on Windows for capture stability
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        except Exception:
            self.cap = cv2.VideoCapture(0)
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
            frame = cv2.flip(frame, 1)
            try:
                if self.frame_q.full():
                    try:
                        self.frame_q.get_nowait()
                    except Empty:
                        pass
                self.frame_q.put_nowait(frame)
            except Exception:
                pass

        try:
            self.cap.release()
        except Exception:
            pass

    # ---------- ROI helper ----------
    def _extract_roi(self, frame, landmarks, w0, h0):
        xs = []; ys = []
        for lm in landmarks.landmark:
            xs.append(int(lm.x * w0))
            ys.append(int(lm.y * h0))
        x_min, x_max = max(min(xs), 0), min(max(xs), w0 - 1)
        y_min, y_max = max(min(ys), 0), min(max(ys), h0 - 1)
        bw, bh = x_max - x_min + 1, y_max - y_min + 1
        mx = int(bw * self.roi_margin)
        my = int(bh * self.roi_margin)
        rx = max(0, x_min - mx)
        ry = max(0, y_min - my)
        rW = min(w0 - rx, bw + 2 * mx)
        rH = min(h0 - ry, bh + 2 * my)
        if rW <= 0 or rH <= 0:
            return frame, (0, 0, w0, h0)
        roi = frame[ry:ry + rH, rx:rx + rW]
        return roi, (rx, ry, rW, rH)

    # ---------- Processing thread ----------
    def process_loop(self):
        last_tick = time.perf_counter()
        window_name = "Finger Circles and Mouse Control"
        last_time = time.perf_counter()

        while not self.stop_event.is_set():
            try:
                frame = self.frame_q.get(timeout=0.05)
            except Empty:
                continue

            self._frame_index += 1
            h0, w0 = frame.shape[:2]

            # decide ROI source
            src_for_det = frame
            roi_info = (0, 0, w0, h0)
            if self.roi_enabled and self.last_landmarks is not None:
                try:
                    roi_img, roi_info = self._extract_roi(frame, self.last_landmarks, w0, h0)
                    if roi_img is not None and roi_img.size > 0:
                        src_for_det = roi_img
                    else:
                        roi_info = (0, 0, w0, h0)
                except Exception:
                    roi_info = (0, 0, w0, h0)

            # resize & convert (GPU optional)
            rgb = self._gpu_resize_cvt(src_for_det, self.proc_width, self.proc_height)

            run_detection = (self._frame_index % self.detection_stride == 0)
            results = None
            if run_detection:
                try:
                    results = self.hands.process(rgb)
                except Exception:
                    results = None

            landmarks = None
            if results and results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                # cache for ROI
                self.last_landmarks = landmarks
                # update last detection time
                self.last_detection_time = time.perf_counter()
            else:
                # if detection skipped, keep last_landmarks for ROI; else if lost, set None
                if run_detection and not results:
                    self.last_landmarks = None

            disp = frame.copy() if self.render_enabled else None
            px, py = None, None
            tip_positions = {}

            # Helper mapping from proc coords to original frame via ROI
            def map_proc_to_orig(cx_p, cy_p):
                rx, ry, rW, rH = roi_info
                cx = int((cx_p / self.proc_width) * rW) + rx
                cy = int((cy_p / self.proc_height) * rH) + ry
                return cx, cy

            if landmarks is not None:
                for lm_id in (4, 8, 12):
                    lm = landmarks.landmark[lm_id]
                    cx_p, cy_p = int(lm.x * self.proc_width), int(lm.y * self.proc_height)
                    cx, cy = map_proc_to_orig(cx_p, cy_p)
                    tip_positions[lm_id] = (cx, cy)
                    if self.render_enabled:
                        color = (0, 255, 255) if lm_id == 4 else ((0, 255, 0) if lm_id == 8 else (255, 0, 255))
                        cv2.circle(disp, (cx, cy), self.radius, color, 2)
                # pointer node
                try:
                    lm_ptr = landmarks.landmark[self.pointer_landmark_id]
                    px_p, py_p = int(lm_ptr.x * self.proc_width), int(lm_ptr.y * self.proc_height)
                    px, py = map_proc_to_orig(px_p, py_p)
                    if self.render_enabled:
                        cv2.circle(disp, (px, py), max(8, self.radius), (0, 0, 255), 2)
                        cv2.circle(disp, (px, py), 3, (0, 0, 255), -1)
                    self.prev_px_py = self.last_px_py
                    self.last_px_py = (px, py)
                except Exception:
                    pass
            else:
                # interpolate if skipping
                if self.prev_px_py and self.last_px_py:
                    t = 1.0 / max(1, self.detection_stride)
                    px = int((1 - t) * self.prev_px_py[0] + t * self.last_px_py[0])
                    py = int((1 - t) * self.prev_px_py[1] + t * self.last_px_py[1])

            # dt for filters
            now_t = time.perf_counter()
            dt = now_t - last_time
            if dt <= 0:
                dt = 1 / 60.0
            last_time = now_t

            # Decide whether we have control
            have_control = self.virtual_mouse_enabled
            if self.release_on_idle:
                # if no hand seen for timeout, release control
                if (time.perf_counter() - self.last_detection_time) * 1000.0 > float(self.idle_timeout_ms):
                    have_control = False
            # If hotkey disabled and virtual mouse disabled => no control already

            # Move mouse if we have control and pointer present
            if have_control and (px is not None and py is not None):
                mouse_x = int(np.interp(px, (0, w0), (0, self.screen_width)))
                mouse_y = int(np.interp(py, (0, h0), (0, self.screen_height)))

                # EWMA smoothing baseline
                if self.smooth_x is None:
                    self.smooth_x, self.smooth_y = mouse_x, mouse_y
                else:
                    a = self.alpha
                    self.smooth_x = int(a * mouse_x + (1 - a) * self.smooth_x)
                    self.smooth_y = int(a * mouse_y + (1 - a) * self.smooth_y)

                out_x, out_y = self.smooth_x, self.smooth_y

                # Kalman
                if self.kalman_enabled:
                    kx, ky = self.kf.correct(self.smooth_x, self.smooth_y, dt)
                    out_x, out_y = int(kx), int(ky)

                # Prediction
                if self.prediction_enabled:
                    lookahead_s = max(0.0, self.prediction_ms / 1000.0)
                    if self.kalman_enabled:
                        pxp, pyp = self.kf.get_predicted_pos(lookahead_s)
                    else:
                        # naive velocity based on last smoothed vs current measurement
                        vx = (self.smooth_x - mouse_x) / max(dt, 1e-3)
                        vy = (self.smooth_y - mouse_y) / max(dt, 1e-3)
                        pxp = out_x + vx * lookahead_s
                        pyp = out_y + vy * lookahead_s
                    out_x = int(np.clip(pxp, 0, self.screen_width - 1))
                    out_y = int(np.clip(pyp, 0, self.screen_height - 1))

                # finally move
                self._move_mouse(out_x, out_y)
            else:
                # not in control: ensure we release buttons so physical mouse works
                self.release_mouse_buttons()

            # Click detection (only if landmarks present)
            if len(tip_positions) >= 3 and have_control:
                ti = np.array(tip_positions.get(4, (0, 0)))
                ii = np.array(tip_positions.get(8, (10**9, 10**9)))
                mi = np.array(tip_positions.get(12, (10**9, 10**9)))
                dist_thumb_index = float(np.linalg.norm(ti - ii))
                dist_thumb_middle = float(np.linalg.norm(ti - mi))
                overlap_thr = 2 * self.radius

                # Left click
                if dist_thumb_index < overlap_thr:
                    self.left_touch_frames += 1
                    self.left_release_frames = 0
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
                    self.right_touch_frames += 1
                    self.right_release_frames = 0
                    if self.right_touch_frames >= self.touch_threshold_frames and not self.right_down:
                        self._mouse_down('right')
                        self.right_down = True
                else:
                    self.right_release_frames += 1
                    if self.right_release_frames >= self.release_threshold_frames and self.right_down:
                        self._mouse_up('right')
                        self.right_down = False
                        self.right_touch_frames = 0
            else:
                # If not have_control or not enough tipped positions -> ensure clicks are released
                # (already covered in release_mouse_buttons but try to ensure counters reset)
                pass

            # Render preview if enabled
            if self.render_enabled:
                try:
                    cv2.imshow(window_name, disp if disp is not None else frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        self.stop_event.set()
                        break
                except Exception:
                    pass

            # throttle to ~60FPS
            now = time.perf_counter()
            dt_proc = now - last_tick
            if dt_proc < (1.0 / 60.0):
                time.sleep((1.0 / 60.0) - dt_proc)
            last_tick = now

        # cleanup
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.release_mouse_buttons()
        try:
            self.hands.close()
        except Exception:
            pass
        try:
            self.root.after(0, self.on_close)
        except Exception:
            pass

    # ---------- Mouse helpers ----------
    def _move_mouse(self, x, y):
        try:
            if USE_PYNPUT_MOUSE:
                self.mouse.position = (x, y)
            else:
                import pyautogui
                pyautogui.moveTo(x, y, _pause=False)
        except Exception:
            pass

    def _mouse_down(self, which):
        try:
            if USE_PYNPUT_MOUSE:
                self.mouse.press(self.btn_left if which == 'left' else self.btn_right)
            else:
                import pyautogui
                pyautogui.mouseDown(button=which)
        except Exception:
            pass

    def _mouse_up(self, which):
        try:
            if USE_PYNPUT_MOUSE:
                self.mouse.release(self.btn_left if which == 'left' else self.btn_right)
            else:
                import pyautogui
                pyautogui.mouseUp(button=which)
        except Exception:
            pass

    def release_mouse_buttons(self):
        try:
            if self.left_down:
                if USE_PYNPUT_MOUSE:
                    self.mouse.release(self.btn_left)
                else:
                    import pyautogui
                    pyautogui.mouseUp(button='left')
                self.left_down = False
        except Exception:
            pass
        try:
            if self.right_down:
                if USE_PYNPUT_MOUSE:
                    self.mouse.release(self.btn_right)
                else:
                    import pyautogui
                    pyautogui.mouseUp(button='right')
                self.right_down = False
        except Exception:
            pass

    # ---------- Shutdown ----------
    def on_close(self):
        # signal threads
        if not self.stop_event.is_set():
            self.stop_event.set()
        # stop hotkey listener if using pynput
        try:
            if hasattr(self, "_hotkey_listener") and self._hotkey_listener is not None:
                try:
                    self._hotkey_listener.stop()
                except Exception:
                    pass
        except Exception:
            pass
        # release camera
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        # destroy windows
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        # close MediaPipe
        try:
            self.hands.close()
        except Exception:
            pass
        # release mouse buttons
        try:
            self.release_mouse_buttons()
        except Exception:
            pass
        # destroy GUI
        try:
            self.root.destroy()
        except Exception:
            try:
                self.root.quit()
            except Exception:
                pass

# ---------------- Run ----------------
if __name__ == "__main__":
    HandGestureMouseControl()
