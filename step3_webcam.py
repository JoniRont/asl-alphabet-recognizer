import cv2
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import time
import numpy as np
import math

# Yhteensopiva Image Processing -tuonti
try:
    from mediapipe.tasks.python.vision import ImageProcessingOptions
except ImportError:
    from mediapipe.tasks.python.components.containers import ImageProcessingOptions

# Määritellään käden liitokset
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

class LandmarkNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.net(x)

CLASS_NAMES = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# LATAUS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load("mediapipe_asl_v1.pth", map_location=device, weights_only=False)
model = LandmarkNet(len(CLASS_NAMES)).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
        num_hands=1
    )
)
img_proc_opts = ImageProcessingOptions(region_of_interest=None, rotation_degrees=0)

# MUUTTUJAT
captured_text = ""
last_stable_label = ""
label_stable_start_time = 0
STABLE_THRESHOLD = 0.8

btn_del = {"rect": (0.02, 0.05, 0.18, 0.18), "start_time": None}
btn_space = {"rect": (0.02, 0.20, 0.18, 0.33), "start_time": None}

motion_buffer = deque(maxlen=40)
prediction_buffer = deque(maxlen=40) # Tallennetaan ennusteet liikkeen ajalta
is_drawing_mode = False
last_movement_time = 0
success_cooldown_until = 0

def check_vertical_alignment(landmarks):
    pinky_base, pinky_tip = landmarks[17], landmarks[20]
    x_diff = abs(pinky_tip.x - pinky_base.x)
    y_diff = pinky_base.y - pinky_tip.y 
    return y_diff > 0.05 and x_diff < (y_diff * 0.4)

def get_palm_direction(landmarks):
    return landmarks[17].x > landmarks[2].x

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    current_time = time.time()
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image, img_proc_opts)

    # Painikkeiden piirto
    for btn, label_name, col in [(btn_del, "DEL", (0,0,255)), (btn_space, "SPACE", (255,0,0))]:
        x1, y1, x2, y2 = btn["rect"]
        color = (0, 255, 0) if btn["start_time"] else col
        cv2.rectangle(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), color, 2)
        cv2.putText(frame, label_name, (int(x1*w)+5, int(y2*h)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if result.hand_landmarks:
        for landmarks in result.hand_landmarks:
            # Piirretään skeleton
            for connection in HAND_CONNECTIONS:
                p1, p2 = landmarks[connection[0]], landmarks[connection[1]]
                cv2.line(frame, (int(p1.x*w), int(p1.y*h)), (int(p2.x*w), int(p2.y*h)), (200, 200, 200), 1)
            for lm in landmarks:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, (0, 255, 0), -1)

            wrist = landmarks[0]
            index_tip = (landmarks[8].x, landmarks[8].y)
            pinky_tip = (landmarks[20].x, landmarks[20].y)
            
            is_vertical = check_vertical_alignment(landmarks)
            is_palm_to_camera = get_palm_direction(landmarks)

            # ENNUSTUS
            lm_list = []
            for lm in landmarks: lm_list.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            input_tensor = torch.FloatTensor(lm_list).to(device).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                conf, predicted = torch.max(torch.softmax(output, dim=1), 1)
                label = CLASS_NAMES[predicted.item()]
                confidence = conf.item()

            display_color = (0, 255, 0)
            status_msg = f"{label} {confidence*100:.0f}%"

            if is_drawing_mode:
                motion_buffer.append(pinky_tip)
                prediction_buffer.append(label)
                
                # Seurataan milloin liike loppuu (pikkurilli pysyy paikallaan)
                # Tai jos kuluu liian kauan (1.5s)
                dist_from_prev = math.sqrt((pinky_tip[0] - motion_buffer[-2][0])**2 + (pinky_tip[1] - motion_buffer[-2][1])**2) if len(motion_buffer) > 1 else 0
                
                if dist_from_prev > 0.005: # Käsi liikkuu edelleen
                    last_movement_time = current_time

                # Jos käsi pysähtyy vähintään 0.5 sekunniksi TAI puskuri täyttyy
                if (current_time - last_movement_time) > 0.5 or len(motion_buffer) == motion_buffer.maxlen:
                    start_p = motion_buffer[0]
                    end_p = motion_buffer[-1]
                    
                    # Lasketaan maksimietäisyys alkupisteestä koko liikkeen ajalta
                    max_dist = 0
                    for p in motion_buffer:
                        d = math.sqrt((p[0] - start_p[0])**2 + (p[1] - start_p[1])**2)
                        if d > max_dist: max_dist = d

                    # TARKISTUS: Oliko liike J vai I?
                    # 1. Jos liike oli suuri JA puskurissa näkyi J-ennusteita
                    if max_dist > 0.08 and "J" in prediction_buffer:
                        captured_text += "j"
                        success_cooldown_until = current_time + 1.0
                    # 2. Jos liike oli olematon ja asento oli I
                    elif max_dist < 0.04 and label == "I" and is_vertical:
                        captured_text += "i"
                        success_cooldown_until = current_time + 0.8
                    
                    is_drawing_mode = False
                    motion_buffer.clear()
                    prediction_buffer.clear()

                # Piirretään liikeviiva
                for i in range(1, len(motion_buffer)):
                    cv2.line(frame, (int(motion_buffer[i-1][0]*w), int(motion_buffer[i-1][1]*h)),
                             (int(motion_buffer[i][0]*w), int(motion_buffer[i][1]*h)), (0, 0, 255), 3)
                status_msg = "ANALYSOIDAAN LIIKETTA..."
                display_color = (0, 0, 255)

            elif current_time > success_cooldown_until:
                # Aloitetaan seuranta heti kun nähdään pysty-I
                if label == 'I' and confidence > 0.80 and is_palm_to_camera and is_vertical:
                    is_drawing_mode = True
                    last_movement_time = current_time
                    motion_buffer.clear()
                    prediction_buffer.clear()
                    motion_buffer.append(pinky_tip)
                    prediction_buffer.append(label)
                
                # Muut kirjaimet (vakaustunnistus)
                elif label not in ["nothing", "del", "space", "I", "J"] and confidence > 0.85:
                    if label == last_stable_label:
                        if (current_time - label_stable_start_time) > STABLE_THRESHOLD:
                            captured_text += label
                            last_stable_label = ""; success_cooldown_until = current_time + 0.8
                    else:
                        last_stable_label = label; label_stable_start_time = current_time

            cv2.putText(frame, status_msg, (int(wrist.x*w), int(wrist.y*h)-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, display_color, 2)

    cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"TEKSTI: {captured_text}", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("ASL Motion Analyzer", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()