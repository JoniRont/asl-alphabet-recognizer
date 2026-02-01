import cv2
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import time

# 1. MODEL STRUCTURE
class LandmarkNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.net(x)

# 2. INITIALIZATION
LABELS = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
          'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load("mediapipe_asl.pth", map_location=device, weights_only=False)
model = LandmarkNet(checkpoint['num_classes']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Setup MediaPipe Hand Landmarker
detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
        num_hands=1
    )
)

# --- VARIABLES ---
captured_text = ""
last_stable_label = ""
label_stable_start_time = 0
STABLE_THRESHOLD = 1.2

# Button definitions (x1, y1, x2, y2) in normalized coordinates for the flipped display
btn_del = {"rect": (0.02, 0.05, 0.20, 0.20), "start_time": None, "triggered": False}
btn_space = {"rect": (0.02, 0.25, 0.20, 0.40), "start_time": None, "triggered": False}

success_cooldown_until = 0

# 3. MAIN LOOP
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    # Mirror the frame for intuitive user experience
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    current_time = time.time()
    
    # Process image for MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)

    # --- DRAW UI BUTTONS ---
    for btn, label_name, base_col in [(btn_del, "DEL", (0,0,255)), (btn_space, "SPACE", (255,0,0))]:
        x1, y1, x2, y2 = btn["rect"]
        color = base_col
        if btn["triggered"]: color = (0, 255, 0) # Green if action performed
        elif btn["start_time"]: color = (0, 255, 255) # Yellow if hovering/loading
        
        cv2.rectangle(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), color, 2)
        cv2.putText(frame, label_name, (int(x1*w)+10, int(y2*h)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if result.hand_landmarks:
        print("Hand detected")
        for landmarks in result.hand_landmarks:
            # MediaPipe detection is on the flipped frame, so coordinates are already in display space
            idx_x, idx_y = landmarks[8].x, landmarks[8].y # Index finger tip

            print(f"Index finger: idx_x={idx_x:.2f}, idx_y={idx_y:.2f}")

            # Visual feedback: purple dot for index finger tip
            cv2.circle(frame, (int(idx_x*w), int(idx_y*h)), 15, (255, 0, 255), -1)

            # --- MODEL INFERENCE ---
            lm_list = [val for lm in landmarks for val in [lm.x, lm.y, lm.z]]
            input_tensor = torch.FloatTensor(lm_list).to(device).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                conf, predicted = torch.max(torch.softmax(output, dim=1), 1)
                label = LABELS[predicted.item()]
                confidence = conf.item()

            # --- BUTTON COLLISION LOGIC ---
            on_any_button = False
            for btn, action in [(btn_del, "del"), (btn_space, "space")]:
                x1, y1, x2, y2 = btn["rect"]
                
                # Check if index finger tip is inside the button
                if x1 < idx_x < x2 and y1 < idx_y < y2:
                    print(f"Finger inside {action} button: idx_x={idx_x:.2f}, idx_y={idx_y:.2f}, rect=({x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f})")
                    on_any_button = True
                    if btn["start_time"] is None:
                        btn["start_time"] = current_time
                        btn["triggered"] = False
                    elif current_time - btn["start_time"] > 0.1 and not btn["triggered"]:
                        print(f"Triggering {action}")
                        if action == "del":
                            captured_text = captured_text[:-1]
                        else:
                            captured_text += " "
                        btn["triggered"] = True # Prevent continuous deletion
                else:
                    # Reset if finger leaves the area
                    btn["start_time"] = None
                    btn["triggered"] = False

            # Stability filter for letter input (disabled when finger is on button)
            if not on_any_button and confidence > 0.85 and current_time > success_cooldown_until:
                if label not in ["nothing", "del", "space", "I", "J", "Z"]:
                    if label == last_stable_label:
                        if (current_time - label_stable_start_time) > STABLE_THRESHOLD:
                            captured_text += label
                            last_stable_label = ""
                            success_cooldown_until = current_time + 0.8
                    else:
                        last_stable_label = label
                        label_stable_start_time = current_time

            # Draw bounding box and label
            x_px = [int(lm.x * w) for lm in landmarks]
            y_px = [int(lm.y * h) for lm in landmarks]
            cv2.rectangle(frame, (min(x_px)-10, min(y_px)-10), (max(x_px)+10, max(y_px)+10), (0, 255, 0), 2)
            # Hide confidence % when finger is in button
            display_text = f"{label}" if on_any_button else f"{label} {confidence:.2f}"
            cv2.putText(frame, display_text, (min(x_px), min(y_px)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # LOWER TEXT BAR
    cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"TEXT: {captured_text}", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("ASL Recognition System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()