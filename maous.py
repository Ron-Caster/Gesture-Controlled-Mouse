import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

class HandMouseController:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Camera dimensions
        self.cam_width, self.cam_height = 640, 480
        
        # Smoothing factor for pointer movement
        self.smoothing = 7
        self.prev_x, self.prev_y = 0, 0
        
        # Click states
        self.left_click_active = False
        self.right_click_active = False
        self.click_cooldown = 0.3  # Cooldown between clicks
        self.last_click_time = 0
        
        # Gesture detection thresholds
        self.finger_tip_threshold = 0.02  # Distance threshold for finger touch detection
        
    def get_finger_positions(self, landmarks):
        """Extract finger tip and relevant joint positions"""
        # Finger landmark indices
        THUMB_TIP = 4
        THUMB_IP = 3
        INDEX_TIP = 8
        INDEX_PIP = 6
        MIDDLE_TIP = 12
        MIDDLE_PIP = 10
        RING_TIP = 16
        RING_PIP = 14
        PINKY_TIP = 20
        
        fingers = {}
        fingers['thumb_tip'] = landmarks[THUMB_TIP]
        fingers['thumb_ip'] = landmarks[THUMB_IP]
        fingers['index_tip'] = landmarks[INDEX_TIP]
        fingers['index_pip'] = landmarks[INDEX_PIP]
        fingers['middle_tip'] = landmarks[MIDDLE_TIP]
        fingers['middle_pip'] = landmarks[MIDDLE_PIP]
        fingers['ring_tip'] = landmarks[RING_TIP]
        fingers['ring_pip'] = landmarks[RING_PIP]
        fingers['pinky_tip'] = landmarks[PINKY_TIP]
        
        return fingers
    
    def is_finger_up(self, tip, pip):
        """Check if a finger is extended (tip above pip)"""
        return tip.y < pip.y
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def detect_gestures(self, fingers):
        """Detect pointing, left click, and right click gestures"""
        # Check if index finger is up (pointing)
        index_up = self.is_finger_up(fingers['index_tip'], fingers['index_pip'])
        
        # Check thumb-middle finger touch for left click
        thumb_middle_distance = self.calculate_distance(fingers['thumb_tip'], fingers['middle_tip'])
        left_click = thumb_middle_distance < self.finger_tip_threshold
        
        # Check thumb-ring finger touch for right click
        thumb_ring_distance = self.calculate_distance(fingers['thumb_tip'], fingers['ring_tip'])
        right_click = thumb_ring_distance < self.finger_tip_threshold
        
        return {
            'pointing': index_up,
            'left_click': left_click,
            'right_click': right_click,
            'pointer_pos': fingers['index_tip']
        }
    
    def smooth_movement(self, x, y):
        """Apply smoothing to mouse movement"""
        smooth_x = self.prev_x + (x - self.prev_x) / self.smoothing
        smooth_y = self.prev_y + (y - self.prev_y) / self.smoothing
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return int(smooth_x), int(smooth_y)
    
    def handle_mouse_actions(self, gestures):
        """Handle mouse pointer movement and clicks"""
        current_time = time.time()
        
        # Move mouse pointer if index finger is pointing
        if gestures['pointing']:
            # Convert normalized coordinates to screen coordinates
            pointer_pos = gestures['pointer_pos']
            screen_x = int(pointer_pos.x * self.screen_width)
            screen_y = int(pointer_pos.y * self.screen_height)
            
            # Apply smoothing
            smooth_x, smooth_y = self.smooth_movement(screen_x, screen_y)
            
            # Move mouse
            pyautogui.moveTo(smooth_x, smooth_y)
        
        # Handle clicks with cooldown
        if current_time - self.last_click_time > self.click_cooldown:
            # Left click (thumb + middle finger)
            if gestures['left_click'] and not self.left_click_active:
                pyautogui.click(button='left')
                self.left_click_active = True
                self.last_click_time = current_time
                print("Left Click")
            elif not gestures['left_click']:
                self.left_click_active = False
            
            # Right click (thumb + ring finger)
            if gestures['right_click'] and not self.right_click_active:
                pyautogui.click(button='right')
                self.right_click_active = True
                self.last_click_time = current_time
                print("Right Click")
            elif not gestures['right_click']:
                self.right_click_active = False
    
    def draw_landmarks_and_gestures(self, image, landmarks, gestures):
        """Draw hand landmarks and gesture indicators on the image"""
        # Draw hand landmarks
        self.mp_draw.draw_landmarks(image, landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        # Draw gesture status
        height, width = image.shape[:2]
        
        # Pointing status
        color = (0, 255, 0) if gestures['pointing'] else (0, 0, 255)
        cv2.putText(image, f"Pointing: {gestures['pointing']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Left click status
        color = (0, 255, 0) if gestures['left_click'] else (0, 0, 255)
        cv2.putText(image, f"Left Click: {gestures['left_click']}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Right click status
        color = (0, 255, 0) if gestures['right_click'] else (0, 0, 255)
        cv2.putText(image, f"Right Click: {gestures['right_click']}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return image
    
    def run(self):
        """Main execution loop"""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        
        print("Hand Mouse Controller Started!")
        print("Gestures:")
        print("- Point with index finger to move mouse")
        print("- Touch thumb + middle finger for left click")
        print("- Touch thumb + ring finger for right click")
        print("- Press 'q' to quit")
        
        # Disable pyautogui failsafe for smooth operation
        pyautogui.FAILSAFE = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process hand detection
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get finger positions
                    fingers = self.get_finger_positions(hand_landmarks.landmark)
                    
                    # Detect gestures
                    gestures = self.detect_gestures(fingers)
                    
                    # Handle mouse actions
                    self.handle_mouse_actions(gestures)
                    
                    # Draw landmarks and gesture status
                    frame = self.draw_landmarks_and_gestures(frame, hand_landmarks, gestures)
            
            # Display the frame
            cv2.imshow('Hand Mouse Controller', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    try:
        controller = HandMouseController()
        controller.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
