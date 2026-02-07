# AI was used to assist in writing this code.
import cv2
import os
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import ImageProcessingOptions
import sys
import time

# SETTINGS
MODEL_PATH = 'hand_landmarker.task'
DATA_DIR = "./../data_1/ASL_Alphabet_Dataset/asl_alphabet_train"
OUTPUT_FILE = "hand_data.csv"

CLASS_NAMES = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize MediaPipe HandLandmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

data_list = []
start_time = time.time()
total_images_processed = 0

print(f"--- Initialize process ---")
print(f"Sourse: {DATA_DIR}")

for idx, label in enumerate(CLASS_NAMES):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.exists(label_path):
        print(f"⚠️ Path not found: {label_path}")
        continue
    
    class_start_time = time.time()
    images = os.listdir(label_path)
    num_images = len(images)
    found_in_class = 0
    
    print(f"\n[{idx+1}/{len(CLASS_NAMES)}] Class: {label} ({num_images} images)")
    
    for count, img_name in enumerate(images):
        img_path = os.path.join(label_path, img_name)
        
        try:
            # Create image
            mp_image = mp.Image.create_from_file(img_path)
            
            image_processing_options = ImageProcessingOptions(
                region_of_interest=None, 
                rotation_degrees=0
            )
            
            result = detector.detect(mp_image, image_processing_options)
            
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                
                # Get wrist (point 0)
                wrist = landmarks[0]
                
                row = []
                for lm in landmarks:
                    # SAVE RELATIVE COORDINATES (lm - wrist)
                    row.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                
                row.append(CLASS_NAMES.index(label))
                data_list.append(row)
                found_in_class += 1
        except Exception as e:
            # Do nothing for now
            continue
        
        # Print progress every 500 images
        if (count + 1) % 500 == 0:
            elapsed = time.time() - start_time
            imgs_per_sec = (total_images_processed + count) / elapsed
            print(f"   > Processed {count+1}/{num_images}... ({imgs_per_sec:.1f} images/s)")

    total_images_processed += num_images
    class_duration = time.time() - class_start_time
    print(f"   ✅ Ready: {label}. Found {found_in_class} hand. Duration: {class_duration:.1f}s")

# Save
print(f"\n--- Save ---")
if data_list:
    columns = []
    for i in range(21):
        columns.extend([f'p{i}_x', f'p{i}_y', f'p{i}_z'])
    columns.append('label')
    df_final = pd.DataFrame(data_list, columns=columns)
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Ready! File '{OUTPUT_FILE}' created. Total {len(data_list)} rows.")
else:
    print("❌ No data to save.")