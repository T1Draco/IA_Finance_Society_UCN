import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# === CONFIGURACIÓN GENERAL ===
ticker = "TSLA"
window_size = 30
epochs = 50
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === RUTAS ===
SCALED_DIR = "../../data/only_close_price_data/scaled"
SPLIT_DIR = "../../data/train_test_split"
SAVE_DIR = "../../results/predictions/lstm"
GRAPH_DIR = "../../results/plots/lstm"
FORECAST_DIR = os.path.join(GRAPH_DIR, "forecast/10d")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(FORECAST_DIR, exist_ok=True)

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

torch.save(model.state_dict(), "models_pytorch/base_lstm_model.pth")

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
print(f"📁 Resultados guardados en: {csv_path}")

# === EVALUACIÓN ===
rmse = np.sqrt(mean_squared_error(df_result["Real"], df_result["Predicho"]))
mae = mean_absolute_error(df_result["Real"], df_result["Predicho"])
r2 = r2_score(df_result["Real"], df_result["Predicho"])
mape = mean_absolute_percentage_error(df_result["Real"], df_result["Predicho"])

print("\n📊 Métricas del Modelo Base:")
print(f"   ✅ RMSE : {rmse:.4f}")
print(f"   ✅ MAE  : {mae:.4f}")
print(f"   ✅ R²   : {r2:.4f}")
print(f"   ✅ MAPE : {mape * 100:.2f}%")

# === GRAFICAR SOLO TEST ===
plt.figure(figsize=(12, 5))
plt.plot(df_result["Date"], df_result["Real"], label="Real", color="blue")
plt.plot(df_result["Date"], df_result["Predicho"], label="Predicción", color="orange", linestyle="--")
plt.title(f"{ticker} – LSTM PyTorch (Solo Test)")
plt.xlabel("Fecha")
plt.ylabel("Precio Close")
plt.legend()
# Texto con métricas
textstr = f"""Métricas:
RMSE  = {rmse:.2f}
MAE   = {mae:.2f}
R²    = {r2:.2f}
MAPE  = {mape*100:.2f}%"""

# Añadir cuadro de texto al gráfico
plt.gca().text(
    0.02, 0.95, textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.8)
)
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(FORECAST_DIR, f"{ticker}_lstm_test_zoom.png")
plt.savefig(plot_path)
plt.close()
print(f"🖼️ Gráfico de test guardado en: {plot_path}")

from datetime import timedelta

# === FORECAST FUTURO ===
forecast_horizon = 10
last_window = test_scaled[-window_size:].reshape(1, window_size, 1)  # Última secuencia
input_seq = torch.tensor(last_window, dtype=torch.float32).to(device)

preds_scaled = []
model.eval()
with torch.no_grad():
    for _ in range(forecast_horizon):
        pred = model(input_seq).cpu().numpy()
        preds_scaled.append(pred[0][0])

        # Desplazar ventana
        input_np = input_seq.cpu().numpy().squeeze()
        new_input = np.append(input_np[1:], pred)
        input_seq = torch.tensor(new_input.reshape(1, window_size, 1), dtype=torch.float32).to(device)

# === Invertir escalado ===
forecast_real = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

# === Fechas futuras ===
last_date = df_result["Date"].iloc[-1]
future_dates = [last_date + timedelta(days=i + 1) for i in range(forecast_horizon)]

# === Calcular Intervalo de Confianza (IC ±1 std del error de test) ===
std_error = np.std(df_result["Real"] - df_result["Predicho"])
lower_bound = forecast_real - std_error
upper_bound = forecast_real + std_error

# === Crear DataFrame de forecast ===
df_forecast = pd.DataFrame({
    "Date": future_dates,
    "Real": [np.nan] * forecast_horizon,
    "Predicho": forecast_real,
    "IC_lower": lower_bound,
    "IC_upper": upper_bound
})

# === Agregar forecast al CSV ===
df_final = pd.concat([df_result, df_forecast], ignore_index=True)
df_final.to_csv(csv_path, index=False)
print(f"📁 Forecast futuro agregado al archivo: {csv_path}")

# === Graficar con IC ===
plt.figure(figsize=(12, 5))
plt.plot(df_result["Date"], df_result["Real"], label="Real", color="blue")
plt.plot(df_final["Date"], df_final["Predicho"], label="Predicción", color="orange", linestyle="--")
plt.fill_between(df_forecast["Date"], lower_bound, upper_bound, color='orange', alpha=0.2, label="IC ±1σ")
plt.axvline(x=last_date, color='gray', linestyle='--', linewidth=1)
plt.title(f"{ticker} – LSTM PyTorch (Test + Forecast)")
plt.xlabel("Fecha")
plt.ylabel("Precio Close")
plt.legend()
plt.grid(True)
plt.tight_layout()
plot_path_forecast = os.path.join(FORECAST_DIR, f"{ticker}_lstm_forecast_10d.png")
plt.savefig(plot_path_forecast)
plt.close()
print(f"🖼️ Gráfico de forecast guardado en: {plot_path_forecast}")
