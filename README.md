🖐️ ASL Real-Time Recognition (MediaPipe & PyTorch)

Tämä projekti toteuttaa amerikkalaisen viittomakielen (ASL) reaaliaikaisen tunnistuksen hyödyntäen MediaPipe Hands -teknologiaa ja PyTorch-neuroverkkoa.
💡 Miksi koordinaatit?

Mallimme muuntaa käden 21 nivelpistettä numeerisiksi koordinaateiksi perinteisen kuvapohjaisen tunnistuksen sijaan.

    Nopeus: Kevyt ja viiveetön myös perusprosessorilla (CPU).

    Varmuus: Immuuni taustahäiriöille ja valaistusmuutoksille.

📁 Projektin rakenne

    step1_create_csv.py: Muuntaa kuvat 63 koordinaatin (21x,y,z) CSV-dataksi.

    step2_train_csv.py: Opettaa MLP-neuroverkon ja tallentaa sen (.pth).

    step3_webcam.py: Live-tunnistus web-kameralla.

✨ Erikoisominaisuudet

    Dynaamiset kirjaimet (J & Z): Tunnistaa liikkeen. Piirrä kuvio ilmassa tai pidä kättä paikallaan 2s valinnan lukitsemiseksi.

    Turva-alueet (Deadzone): Tunnistus pysähtyy, kun käsi on del tai space -painikkeiden päällä.

    Vakaussuodatin: Kirjain lisätään vasta, kun asento pysyy vakaana 1.2s.

🚀 Käyttöohjeet
1. Lataa opetusdata

Lataa kuvadata tästä osoitteesta:

🔗 Kaggle: ASL American Sign Language Alphabet Dataset

    [!IMPORTANT] Tärkeää: Varmista, että koodin CLASS_NAMES -listan järjestys on täsmälleen sama kuin lataamasi datan kansiojärjestys levyllä.

2. Suoritusjärjestys

    python step1_create_csv.py (Datan esikäsittely)

    python step2_train_csv.py (Mallin opetus)

    python step3_webcam.py (Live-tunnistus)

🖐️ ASL Real-Time Recognition (English)

Real-time American Sign Language (ASL) recognition using MediaPipe Hands and a PyTorch neural network.
💡 Why Landmarks?

Instead of raw pixels, we process 21 hand joints as numerical coordinates.

    Efficiency: Extremely fast; runs smoothly on CPU.

    Robustness: Immune to background noise and lighting conditions.

🚀 Getting Started
1. Download Dataset

Download the training images from:

🔗 Kaggle: ASL American Sign Language Alphabet Dataset

    [!IMPORTANT] Crucial: Ensure that the CLASS_NAMES list in the scripts matches the exact order of the folders in your local dataset directory.

2. Execution Steps

    python step1_create_csv.py (Preprocessing)

    python step2_train_csv.py (Training)

    python step3_webcam.py (Live Detection)