import cv2
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import time

# 1. MALLIN RAKENNE
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

# 2. ALUSTUS
LABELS = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
          'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load("mediapipe_asl.pth", map_location=device, weights_only=False)
model = LandmarkNet(checkpoint['num_classes']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
        num_hands=1
    )
)

# --- MUUTTUJAT ---
captured_text = ""
last_stable_label = ""
label_stable_start_time = 0
STABLE_THRESHOLD = 1.2

btn_del = {"rect": (0.02, 0.05, 0.18, 0.18), "start_time": None}
btn_space = {"rect": (0.02, 0.20, 0.18, 0.33), "start_time": None}

motion_buffer = deque(maxlen=60)
is_drawing_mode = False
drawing_target = "" 
last_movement_time = 0
success_cooldown_until = 0

def detect_strict_j_motion(buffer):
    if len(buffer) < 15: return False
    y_coords = [p[1] for p in buffer]
    x_coords = [p[0] for p in buffer]
    # Ylhäältä alas: loppu-y miinus alku-y on positiivinen
    vertical_drop = y_coords[-1] - y_coords[0]
    # Koukku: sormi liikkuu sivusuunnassa
    has_hook = abs(x_coords[-1] - x_coords[0]) > 0.04
    return vertical_drop > 0.15 and has_hook

# 3. PÄÄSILMUKKA
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    current_time = time.time()
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)

    # Painikkeiden piirto (aina näkyvissä)
    for btn, label_name, col in [(btn_del, "del", (0,0,255)), (btn_space, "space", (255,0,0))]:
        x1, y1, x2, y2 = btn["rect"]
        draw_col = (0, 255, 0) if btn["start_time"] else col
        cv2.rectangle(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), draw_col, 2)
        cv2.putText(frame, label_name, (int(x1*w)+5, int(y2*h)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_col, 1)

    if result.hand_landmarks:
        for landmarks in result.hand_landmarks:
            lm_list = [val for lm in landmarks for val in [lm.x, lm.y, lm.z]]
            index_tip = (landmarks[8].x, landmarks[8].y)
            middle_tip = (landmarks[12].x, landmarks[12].y)
            pinky_tip = (landmarks[20].x, landmarks[20].y)

            # --- 1. PAINIKKEIDEN LOGIIKKA (KORJATTU) ---
            on_button = False
            for btn, action in [(btn_del, "del"), (btn_space, "space")]:
                x1, y1, x2, y2 = btn["rect"]
                # Tarkistetaan onko etusormi JA keskisormi napin päällä
                if (x1 < index_tip[0] < x2 and y1 < index_tip[1] < y2) and \
                   (x1 < middle_tip[0] < x2 and y1 < middle_tip[1] < y2):
                    on_button = True
                    if btn["start_time"] is None:
                        btn["start_time"] = current_time
                    elif current_time - btn["start_time"] > 1.0:
                        if action == "del": captured_text = captured_text[:-1]
                        else: captured_text += " "
                        btn["start_time"] = current_time + 1.2 # Estä sarjatuli
                else:
                    btn["start_time"] = None

            if on_button:
                is_drawing_mode = False; motion_buffer.clear()
                continue

            # --- 2. MALLIN ENNUSTUS ---
            input_tensor = torch.FloatTensor(lm_list).to(device).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
                conf, predicted = torch.max(torch.softmax(output, dim=1), 1)
                label = LABELS[predicted.item()]
                confidence = conf.item()

            in_success_pause = current_time < success_cooldown_until
            color = (0, 255, 0)
            display_label = f"{label} {confidence*100:.0f}%"

            # --- 3. J-LOGIIKKA (TIUKKA LIIKE) ---
            if is_drawing_mode:
                motion_buffer.append(pinky_tip)
                motion_ok = detect_strict_j_motion(motion_buffer)
                at_final_pose = (label == "J" and confidence > 0.75)
                
                color = (0, 0, 255)
                display_label = "LIIKE OK! NAYTA J" if motion_ok else "PIIRRA J ALAS"

                if motion_ok and at_final_pose:
                    captured_text += "j"
                    is_drawing_mode = False; motion_buffer.clear()
                    success_cooldown_until = current_time + 1.2
                elif (current_time - last_movement_time) > 2.0:
                    # Jos ei liikettä, katsotaan oliko kyseessä vain 'i'
                    if (pinky_tip[1] - motion_buffer[0][1]) < 0.05: captured_text += "i"
                    is_drawing_mode = False; motion_buffer.clear()
                    success_cooldown_until = current_time + 1.2
            
            elif not in_success_pause:
                if label == 'I' and confidence > 0.85:
                    is_drawing_mode = True; drawing_target = "J"
                    last_movement_time = current_time
                    motion_buffer.clear(); motion_buffer.append(pinky_tip)
                elif label not in ["nothing", "del", "space", "I", "J", "Z"] and confidence > 0.75:
                    if label == last_stable_label:
                        if (current_time - label_stable_start_time) > STABLE_THRESHOLD:
                            captured_text += label
                            last_stable_label = ""; label_stable_start_time = current_time + 1.0
                    else:
                        last_stable_label = label; label_stable_start_time = current_time

            # Visualisointi
            for i in range(1, len(motion_buffer)):
                cv2.line(frame, (int(motion_buffer[i-1][0]*w), int(motion_buffer[i-1][1]*h)),
                         (int(motion_buffer[i][0]*w), int(motion_buffer[i][1]*h)), (255, 255, 0), 2)
            
            x_coords_px = [int(lm.x * w) for lm in landmarks]
            y_coords_px = [int(lm.y * h) for lm in landmarks]
            cv2.rectangle(frame, (min(x_coords_px)-20, min(y_coords_px)-20), (max(x_coords_px)+20, max(y_coords_px)+20), color, 2)
            cv2.putText(frame, display_label, (min(x_coords_px), min(y_coords_px)-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Alapalkki
    cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"TEKSTI: {captured_text}", (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("ASL Writer - Fixed Buttons & J", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()