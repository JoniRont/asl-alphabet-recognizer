import cv2
import os
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# KORJATTU IMPORT: ImageProcessingOptions löytyy nykyään täältä:
from mediapipe.tasks.python.vision import ImageProcessingOptions
import sys
import time

# ASETUKSET
MODEL_PATH = 'hand_landmarker.task'
# Varmista että tämä polku on oikein suhteessa skriptiin
DATA_DIR = "./../data_1/ASL_Alphabet_Dataset/asl_alphabet_train"
OUTPUT_FILE = "hand_data.csv"

CLASS_NAMES = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Alustus
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

data_list = []
start_time = time.time()
total_images_processed = 0

print(f"--- ALOITETAAN PROSESSOINTI ---")
print(f"Lähde: {DATA_DIR}")

for idx, label in enumerate(CLASS_NAMES):
    label_path = os.path.join(DATA_DIR, label)
    if not os.path.exists(label_path):
        print(f"⚠️ Polkua ei löydy: {label_path}")
        continue
    
    class_start_time = time.time()
    images = os.listdir(label_path)
    num_images = len(images)
    found_in_class = 0
    
    print(f"\n[{idx+1}/{len(CLASS_NAMES)}] Luokka: {label} ({num_images} kuvaa)")
    
    for count, img_name in enumerate(images):
        img_path = os.path.join(label_path, img_name)
        
        try:
            # Luodaan kuva
            mp_image = mp.Image.create_from_file(img_path)
            
            # KORJAUS: Käytetään oikeaa polkua ImageProcessingOptionsille
            image_processing_options = ImageProcessingOptions(
                region_of_interest=None, 
                rotation_degrees=0
            )
            
            result = detector.detect(mp_image, image_processing_options)
            
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                
                # HAETAAN RANNE (piste 0) normalisointia varten
                wrist = landmarks[0]
                
                row = []
                for lm in landmarks:
                    # TALLENNETAAN SUHTEELLISET KOORDINAATIT (lm - wrist)
                    row.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                
                row.append(CLASS_NAMES.index(label))
                data_list.append(row)
                found_in_class += 1
        except Exception as e:
            # Jos virhe on kriittinen, tulosta se (vapaaehtoinen)
            # print(f"Virhe kuvassa {img_name}: {e}")
            continue
        
        # Tulostetaan edistyminen
        if (count + 1) % 500 == 0:
            elapsed = time.time() - start_time
            # Käytetään count-muuttujaa laskentaan
            imgs_per_sec = (total_images_processed + count) / elapsed
            print(f"   > Käsitelty {count+1}/{num_images}... ({imgs_per_sec:.1f} kuvaa/s)")

    total_images_processed += num_images
    class_duration = time.time() - class_start_time
    print(f"   ✅ Valmis: {label}. Löydetty {found_in_class} kättä. Kesto: {class_duration:.1f}s")

# Tallennus
print(f"\n--- TALLENNETAAN ---")
if data_list:
    columns = []
    for i in range(21):
        columns.extend([f'p{i}_x', f'p{i}_y', f'p{i}_z'])
    columns.append('label')
    df_final = pd.DataFrame(data_list, columns=columns)
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ VALMIS! Tiedosto '{OUTPUT_FILE}' luotu. Yhteensä {len(data_list)} riviä.")
else:
    print("❌ Ei dataa tallennettavaksi.")