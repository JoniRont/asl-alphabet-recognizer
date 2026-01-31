🖐️ ASL Real-Time Recognition (MediaPipe & PyTorch)

Tämä projekti toteuttaa amerikkalaisen viittomakielen (ASL) reaaliaikaisen tunnistuksen hyödyntäen MediaPipe Hands -teknologiaa ja PyTorch-neuroverkkoa.

Perinteisen kuvapohjaisen tunnistuksen sijaan tämä malli muuntaa käden 21 nivelpistettä numeerisiksi koordinaateiksi. Tämä tekee sovelluksesta erittäin nopean, kevyen ja immuunin taustahäiriöille, kuten varjoille tai monimutkaisille huoneympäristöille.
📋 Sisällysluettelo

    Vaatimukset

    Asennus

    Projektin rakenne

    Käyttöohjeet

        Vaihe 1: Datan esikäsittely

        Vaihe 2: Mallin opetus

        Vaihe 3: Live-tunnistus

    Miten se toimii?

💻 Vaatimukset

    Python 3.8 tai uudempi

    Web-kamera

    (Valinnainen) NVIDIA GPU ja CUDA, jos haluat nopeuttaa opetusta (vaikka tämä malli on kevyt myös CPU:lla).

🛠 Asennus

    Lataa projekti ja siirry projektikansioon.

    Luo virtuaaliympäristö (suositus):
    Bash

    python -m venv venv
    source venv/bin/activate  # Mac/Linux
    .\venv\Scripts\activate   # Windows

    Asenna tarvittavat kirjastot:
    Bash

    pip install -r requirements.txt

📁 Projektin rakenne
Plaintext

├── data_1/                  # Alkuperäiset opetuskuvat (A-Z, del, jne.)
├── step1_create_csv.py      # Muuntaa kuvat koordinaateiksi (CSV)
├── step2_train_csv.py       # Opettaa neuroverkon CSV-datalla
├── step3_webcam.py          # Reaaliaikainen tunnistusohjelma
├── hand_data.csv            # Generoitu koordinaattidata (luodaan vaiheessa 1)
├── mediapipe_asl.pth        # Opetettu mallitiedosto (luodaan vaiheessa 2)
├── requirements.txt         # Kirjastoluettelo
└── README.md                # Tämä ohjetiedosto

🚀 Käyttöohjeet
Vaihe 1: Datan esikäsittely (step1_create_csv.py)

Tämä skripti lukee data_1-kansion kuvat ja käyttää MediaPipea etsimään käden nivelpisteet.

    Toiminto: Luo hand_data.csv -tiedoston, joka sisältää 63 koordinaattia (21 nivelta¨×x,y,z) per kuva.

    Ajo: python step1_create_csv.py

Vaihe 2: Mallin opetus (step2_train_csv.py)

Opetetaan kevyt Multi-Layer Perceptron (MLP) -neuroverkko tunnistamaan kirjaimet koordinaattien perusteella.

    Toiminto: Lukee CSV-tiedoston ja tallentaa opitun mallin nimellä mediapipe_asl.pth.

    Ajo: python step2_train_csv.py

Vaihe 3: Live-tunnistus (step3_webcam.py)

Käynnistää webkameran ja suorittaa tunnistuksen livenä.

    Toiminto: Piirtää käden päälle "luurangon" ja näyttää ennustetun kirjaimen sekä varmuusprosentin.

    Ajo: python step3_webcam.py

    Lopetus: Paina näppäimistöstä 'q'.

🧠 Miten se toimii?

Järjestelmä on jaettu kolmeen älykkääseen kerrokseen:

    MediaPipe Hands: Google MediaPipe tunnistaa käden kuvasta ja erottaa siitä 21 avainpistettä. Tämä poistaa tarpeen analysoida taustaa tai värejä.

    Normalisointi: Koordinaatit käsitellään niin, että käden etäisyys kamerasta tai sijainti ruudulla ei vaikuta lopputulokseen.

    PyTorch Classifier: Syväoppiva malli saa syötteeksi nivelten asennot ja luokittelee ne oikeaksi ASL-merkiksi.

⚠️ Huomioitavaa

    Varmista, että CLASS_NAMES -lista on identtinen kaikissa kolmessa skriptissä.

    Jos malli ei tunnista kättäsi, varmista hyvä valaistus, jotta MediaPipe löytää nivelpisteet oikein.