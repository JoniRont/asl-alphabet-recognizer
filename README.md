🖐️ ASL Real-Time Recognition (MediaPipe & PyTorch)

Real-time American Sign Language (ASL) recognition using MediaPipe Hands and a PyTorch neural network.

💡 Why Landmarks?

Instead of raw pixels, we process 21 hand joints as numerical coordinates.

    Efficiency: Extremely fast; runs smoothly on CPU.

    Robustness: Immune to background noise and lighting conditions.

📁 Project Structure

    step1_create_csv.py: Converts images to 63 coordinate (21 x,y,z) CSV data.

    step2_train_csv.py: Trains an MLP neural network and saves it (.pth).

    step3_webcam.py: Live detection with webcam.

✨ Special Features

    Dynamic Letters (J & Z): Recognizes movement. Draw a pattern in the air or hold your hand steady for 2 seconds to lock in the selection.

    Safety Zones (Deadzone): Recognition stops when your hand is over the del or space buttons.

    Stability Filter: A letter is added only after the gesture stays stable for 1.2 seconds.

🚀 Getting Started

1. Download Dataset

Download the training images from:

🔗 Kaggle: ASL American Sign Language Alphabet Dataset

    [!IMPORTANT] Crucial: Ensure that the CLASS_NAMES list in the scripts matches the exact order of the folders in your local dataset directory.

2. Execution Steps

    python step1_create_csv.py (Preprocessing)

    python step2_train_csv.py (Training)

    python step3_webcam.py (Live Detection)