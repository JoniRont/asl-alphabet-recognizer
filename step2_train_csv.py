import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# 1. LAITEVALINTA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n--- ALOITETAAN PARANNETTU OPETUS ---")
print(f"Käytettävä laite: {device.type.upper()}")

# 2. PARANNETTU MALLIN MÄÄRITTELY (Batch Normalization lisätty)
class LandmarkNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 128),
            nn.BatchNorm1d(128), # Vakauttaa opetusta ja nopeuttaa konvergenssia
            nn.ReLU(),
            nn.Dropout(0.2),     # Estää ylisovitusta
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, num_classes)
        )
    def forward(self, x): 
        return self.net(x)

# 3. DATAN LATAUS JA ESIKÄSITTELY
def preprocess_data(X):
    """
    Normalisoi koordinaatit ranteen (landmark 0) suhteen.
    Oletetaan että data on muodossa [x0, y0, z0, x1, y1, z1, ...]
    """
    X_norm = X.copy()
    for i in range(len(X_norm)):
        # Haetaan ranteen koordinaatit (ensimmäiset 3 arvoa)
        wrist_x, wrist_y, wrist_z = X_norm[i, 0], X_norm[i, 1], X_norm[i, 2]
        # Vähennetään ranteen sijainti kaikista pisteistä
        for j in range(0, 63, 3):
            X_norm[i, j] -= wrist_x
            X_norm[i, j+1] -= wrist_y
            X_norm[i, j+2] -= wrist_z
    return X_norm

load_start = time.time()
df = pd.read_csv('hand_data.csv')
print(f"CSV ladattu ({time.time()-load_start:.1f}s)")

X = df.drop('label', axis=1).values
y = df['label'].values

# Suoritetaan normalisointi ranteen suhteen
print("Normalisoidaan koordinaatit ranteen suhteen...")
X = preprocess_data(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Muunnos tensoreiksi
X_train_t = torch.FloatTensor(X_train).to(device)
y_train_t = torch.LongTensor(y_train).to(device)
X_test_t = torch.FloatTensor(X_test).to(device)
y_test_t = torch.LongTensor(y_test).to(device)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

# 4. ALUSTUS
num_classes = len(np.unique(y))
model = LandmarkNet(num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 5. OPETUSSYKLI EARLY STOPPINGILLA
epochs = 200
patience = 15  # Kuinka monta epochia odotetaan ilman parannusta
best_accuracy = 0
early_stop_counter = 0

print(f"Datan koko: {len(X_train)} näytettä | Luokkia: {num_classes}")
print("-" * 50)

start_time = time.time()

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Validointi jokaisen epochin jälkeen
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test_t).sum().item() / y_test_t.size(0)
    
    # Early stopping ja parhaan tallennus
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        early_stop_counter = 0
        # Tallennetaan paras tilapäisesti muistiin
        best_model_state = model.state_dict()
        status_msg = f"--> Uusi paras! Accuracy: {accuracy*100:.2f}%"
    else:
        early_stop_counter += 1
        status_msg = f"Accuracy: {accuracy*100:.2f}%"

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | {status_msg}")

    if early_stop_counter >= patience:
        print(f"\n[Early Stopping] Opetus lopetettu epochilla {epoch+1}. Malli ei enää parantunut.")
        break

# Ladattava paras malli ennen tallennusta
model.load_state_dict(best_model_state)

total_duration = time.time() - start_time
print("-" * 50)
print(f"Opetus valmis! Paras tarkkuus: {best_accuracy*100:.2f}%")
print(f"Kokonaiskesto: {total_duration:.1f} sekuntia.")

# 6. TALLENNUS (Sisältää myös luokat)
class_names = sorted(df['label'].unique().tolist())
torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': num_classes,
    'classes': class_names
}, "mediapipe_asl_v1.pth")

print(f"[OK] Malli tallennettu: mediapipe_asl_v1.pth")