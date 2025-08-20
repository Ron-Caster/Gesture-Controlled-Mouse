import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
import threading

class HandCircleVisualizer:
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
        self.radius = 20  # Starting pixel radius

        # Tkinter GUI
        self.root = tk.Tk()
        self.root.title("Circle Radius Adjuster")
        self.root.geometry("300x100")

        ttk.Label(self.root, text="Adjust Circle Radius (1-100):").pack(pady=10)

        self.radius_slider = ttk.Scale(self.root, from_=1, to=100, orient='horizontal',
                                       command=self.update_radius)
        self.radius_slider.set(self.radius)
        self.radius_slider.pack(fill='x', padx=10)

        # Start GUI in main thread
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

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Fingertip landmarks: thumb (4), index (8), middle (12)
                fingertips = [4, 8, 12]
                colors = [(0, 255, 255), (0, 255, 0), (255, 0, 255)]  # Yellow, Green, Purple

                h, w = frame.shape[:2]
                for idx, lm_id in enumerate(fingertips):
                    lm = hand_landmarks.landmark[lm_id]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), self.radius, colors[idx], 2)

            cv2.imshow("Finger Circles", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.root.quit()  # Close GUI when video loop ends

if __name__ == "__main__":
    HandCircleVisualizer()
