# AI was used to assist in writing this code.
import cv2
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import time
import math

# Compatibility imports for different Mediapipe versions
try:
    from mediapipe.tasks.python.vision import ImageProcessingOptions
except ImportError:
    from mediapipe.tasks.python.components.containers import ImageProcessingOptions

# Define the connections between hand landmarks for visualization
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]

# Define the neural network architecture for landmark classification
class LandmarkNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.net(x)

# List of ASL alphabet classes and gestures including special functions
CLASS_NAMES = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# List of labels that are ignored during drawing to prevent accidental resets
IGNORE_RESET_LABELS = ["Z", "I", "J", "H", "D", "U", "V", "G", "Q", "P", "nothing"]

# Initialize device, load the trained model checkpoint, and set to evaluation mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load("mediapipe_asl_v1.pth", map_location=device, weights_only=False)
model = LandmarkNet(len(CLASS_NAMES)).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Set up Mediapipe Hand Landmarker with the pre-trained task file
detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
        num_hands=1
    )
)
img_proc_opts = ImageProcessingOptions(region_of_interest=None, rotation_degrees=0)

# Initialize variables for text capturing, stability tracking, and drawing buffers
captured_text = ""
last_stable_label = ""
label_stable_start_time = 0
STABLE_THRESHOLD = 0.8
motion_buffer = deque(maxlen=100)
prediction_buffer = deque(maxlen=100)
is_drawing_mode = False
drawing_type = None 
last_movement_time = 0
success_cooldown_until = 0 
debug_msg = ""

# Variables for handling gesture cancellation logic
cancel_label = ""
cancel_start_time = 0

# Define UI button areas (relative coordinates) and their interaction timers
btn_del = {"rect": (0.02, 0.05, 0.18, 0.18), "start_time": None}
btn_space = {"rect": (0.02, 0.20, 0.18, 0.33), "start_time": None}

# Helper function to check if a specific finger is extended based on Y-coordinates
def is_finger_extended(landmarks, tip_idx):
    return landmarks[tip_idx].y < landmarks[tip_idx - 2].y

# Initialize webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1) # Flip frame horizontally for a mirror effect
    h, w, _ = frame.shape
    current_time = time.time()
    
    # Process the frame with Mediapipe to detect hand landmarks
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image, img_proc_opts)

    # Draw UI buttons (DEL and SPACE) and update their visual state
    for btn, label_name, col in [(btn_del, "DEL", (0,0,255)), (btn_space, "SPACE", (255,0,0))]:
        x1, y1, x2, y2 = btn["rect"]
        color = (0, 255, 0) if btn["start_time"] else col
        cv2.rectangle(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), color, 2)
        cv2.putText(frame, label_name, (int(x1*w)+5, int(y2*h)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if result.hand_landmarks:
        for landmarks in result.hand_landmarks:
            # Extract key landmark positions and finger states
            wrist = landmarks[0]
            index_tip = (landmarks[8].x, landmarks[8].y)
            index_extended = is_finger_extended(landmarks, 8)
            pinky_extended = is_finger_extended(landmarks, 20)

            # Check for collisions between index finger and UI buttons
            on_any_button = False
            for btn, action in [(btn_del, "del"), (btn_space, "space")]:
                x1, y1, x2, y2 = btn["rect"]
                if x1 < index_tip[0] < x2 and y1 < index_tip[1] < y2:
                    on_any_button = True
                    if is_drawing_mode:
                        is_drawing_mode = False; motion_buffer.clear()
                        debug_msg = ""
                    
                    if btn["start_time"] is None: btn["start_time"] = current_time
                    elif current_time - btn["start_time"] > 0.1:
                        captured_text = captured_text[:-1] if action == "del" else captured_text + " "
                        btn["start_time"] = current_time + 1000 
                        success_cooldown_until = current_time + 1.2
                else: btn["start_time"] = None

            # Prepare landmark data and run prediction through the neural network
            lm_list = []
            for lm in landmarks: lm_list.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            input_tensor = torch.FloatTensor(lm_list).to(device).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                conf, _ = torch.max(torch.softmax(output, dim=1), 1)
                label = CLASS_NAMES[torch.argmax(output).item()]
                confidence = conf.item()

            # Handle logic when the user is actively drawing a gesture (Z or J)
            if is_drawing_mode:
                # Cancel drawing if a non-compatible gesture is held for more than 0.2 seconds
                if (label not in IGNORE_RESET_LABELS) and confidence > 0.90:
                    if cancel_label != label:
                        cancel_label = label; cancel_start_time = current_time
                    elif current_time - cancel_start_time > 0.2:
                        is_drawing_mode = False; motion_buffer.clear()
                        debug_msg = f"Cancelled ({label})"
                else:
                    cancel_label = ""; cancel_start_time = 0
                    active_point = index_tip if drawing_type == "Z" else (landmarks[20].x, landmarks[20].y)
                    motion_buffer.append(active_point)
                    prediction_buffer.append(label)
                    
                    # Track movement to detect when the drawing stroke has ended
                    if len(motion_buffer) > 1:
                        dist = math.sqrt((active_point[0]-motion_buffer[-2][0])**2 + (active_point[1]-motion_buffer[-2][1])**2)
                        if dist > 0.005: last_movement_time = current_time

                    # Finalize the gesture once movement stops or the buffer is full
                    if (current_time - last_movement_time) > 0.7 or len(motion_buffer) == motion_buffer.maxlen:
                        if drawing_type == "Z" and len(motion_buffer) > 15:
                            xs = [p[0] for p in motion_buffer]
                            max_x_idx = xs.index(max(xs))
                            # Validate Z-shape: requires sufficient width and a diagonal stroke back
                            if (max(xs) - min(xs)) > 0.04 and max_x_idx < len(xs) * 0.7:
                                if (max(xs) - min(xs[max_x_idx:])) > 0.03:
                                    captured_text += "z"
                                    debug_msg = "" 
                                else: debug_msg = "ERROR: Diagonal stroke missing"
                            else: debug_msg = "ERROR: Too straight"
                        
                        elif drawing_type == "I/J":
                            # Distinguish between static 'i' and dynamic 'j' based on movement distance
                            max_dist = max([math.sqrt((p[0]-motion_buffer[0][0])**2 + (p[1]-motion_buffer[0][1])**2) for p in motion_buffer])
                            if max_dist > 0.08 and "J" in prediction_buffer: captured_text += "j"
                            elif max_dist < 0.04 and label == "I": captured_text += "i"
                            debug_msg = ""
                        
                        is_drawing_mode = False; success_cooldown_until = current_time + 1.2
                        motion_buffer.clear()

                # Visualize the drawing path on the screen
                for i in range(1, len(motion_buffer)):
                    cv2.line(frame, (int(motion_buffer[i-1][0]*w), int(motion_buffer[i-1][1]*h)),
                             (int(motion_buffer[i][0]*w), int(motion_buffer[i][1]*h)), (0, 0, 255), 3)

            # Handle detection of static gestures and initiation of drawing modes
            elif current_time > success_cooldown_until:
                if label == 'Z' and confidence > 0.80 and index_extended and not on_any_button:
                    is_drawing_mode, drawing_type, last_movement_time = True, "Z", current_time
                    motion_buffer.clear(); debug_msg = "Z..."
                elif label == 'I' and confidence > 0.80 and pinky_extended and not index_extended and not on_any_button:
                    is_drawing_mode, drawing_type, last_movement_time = True, "I/J", current_time
                    motion_buffer.clear(); debug_msg = "I/J..."
                elif label not in ["nothing", "del", "space", "I", "J", "Z"] and confidence > 0.85 and not on_any_button:
                    # Require the gesture to be held stable for a specific duration before registering
                    if label == last_stable_label:
                        if (current_time - label_stable_start_time) > STABLE_THRESHOLD:
                            captured_text += label
                            last_stable_label = ""; success_cooldown_until = current_time + 1.2
                            debug_msg = ""
                    else:
                        last_stable_label = label; label_stable_start_time = current_time

            # Display current prediction and debug messages on the frame
            cv2.putText(frame, f"{label} {confidence*100:.0f}%", (int(wrist.x*w), int(wrist.y*h)-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, debug_msg, (w//2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Render the captured text box at the bottom of the screen
    cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"TEXT: {captured_text}", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("ASL Analyzer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# Clean up resources
cap.release()
cv2.destroyAllWindows()