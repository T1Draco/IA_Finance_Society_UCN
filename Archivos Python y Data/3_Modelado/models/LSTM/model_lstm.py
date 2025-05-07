import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error

# === CONFIGURACIÓN GENERAL ===
ticker = "TSLA"
window_size = 30
epochs = 50
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === RUTAS ===
SCALED_DIR = "../../data/scaled"
SPLIT_DIR = "../../data/train_test_split"
SAVE_DIR = "../../results/predictions"
GRAPH_DIR = "../../results/plots"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# === CARGA DE DATOS ESCALADOS ===
train_scaled = np.load(os.path.join(SCALED_DIR, f"{ticker}_train_scaled.npy"))
test_scaled = np.load(os.path.join(SCALED_DIR, f"{ticker}_test_scaled.npy"))
scaler = joblib.load(os.path.join(SCALED_DIR, f"{ticker}_scaler.pkl"))

df_test = pd.read_csv(os.path.join(SPLIT_DIR, f"{ticker}_test.csv"), parse_dates=["Date"])
df_train = pd.read_csv(os.path.join(SPLIT_DIR, f"{ticker}_train.csv"), parse_dates=["Date"])

# === CREAR SECUENCIAS ===
def crear_secuencias(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X_train, y_train = crear_secuencias(train_scaled, window_size)
X_test, y_test = crear_secuencias(test_scaled, window_size)

# Convertir a tensores
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)

# === DEFINICIÓN DEL MODELO LSTM ===
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Último output temporal
        return out

model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === ENTRENAMIENTO ===
model.train()
for epoch in range(epochs):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch).squeeze()
        loss = criterion(output, y_batch.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f" Época {epoch+1}/{epochs} – Pérdida: {total_loss:.4f}")

# === PREDICCIÓN ===
model.eval()
with torch.no_grad():
    pred_scaled = model(X_test).cpu().numpy()

real = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))
pred = scaler.inverse_transform(pred_scaled)

# === GUARDAR RESULTADOS ===
fechas_test = df_test["Date"].iloc[window_size:].reset_index(drop=True)
df_result = pd.DataFrame({
    "Date": fechas_test,
    "Real": real.flatten(),
    "Predicho": pred.flatten()
})
csv_path = os.path.join(SAVE_DIR, f"{ticker}_lstm_pytorch_predicciones.csv")
df_result.to_csv(csv_path, index=False)
print(f" Resultados guardados en: {csv_path}")

# === EVALUACIÓN ===
rmse = np.sqrt(mean_squared_error(df_result["Real"], df_result["Predicho"]))
mae = mean_absolute_error(df_result["Real"], df_result["Predicho"])
print(f"📊 RMSE: {rmse:.2f} | MAE: {mae:.2f}")

# === GRAFICAR ===
plt.figure(figsize=(12, 5))
plt.plot(df_train["Date"], df_train["Close"], label="Train", color="gray", alpha=0.6)
plt.plot(df_result["Date"], df_result["Real"], label="Test Real", color="blue")
plt.plot(df_result["Date"], df_result["Predicho"], label="Predicción LSTM (PyTorch)", color="orange", linestyle="--")
plt.title(f"{ticker} – LSTM PyTorch (ventana={window_size})")
plt.xlabel("Fecha")
plt.ylabel("Precio Close")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, f"{ticker}_lstm_pytorch_forecast.png"))
plt.close()
print(f" Gráfico guardado: {ticker}_lstm_pytorch_forecast.png")
