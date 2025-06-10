import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from datetime import timedelta

# === CONFIGURACIÓN GENERAL ===
ticker = "CL=F"
window_size = 60
epochs = 60
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === RUTAS ===
SCALED_DIR = "../../data/multivariable_data/multivariable_scaled"
SPLIT_DIR = "../../data/train_test_split"
SAVE_DIR = "../../results/predictions/lstm_multivariable"
GRAPH_DIR = "../../results/plots/lstm_multivariable"
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

# === CREAR SECUENCIAS MULTIVARIABLES ===
def crear_secuencias(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i][0])  # Close es la primera columna
    return np.array(X), np.array(y)

X_train, y_train = crear_secuencias(train_scaled, window_size)
X_test, y_test = crear_secuencias(test_scaled, window_size)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

# === DEFINICIÓN DEL MODELO MULTIVARIABLE ===
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
input_size = train_scaled.shape[1]
model = LSTMModel(input_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses = []

# === ENTRENAMIENTO ===
model.train()
for epoch in range(epochs):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch).squeeze()
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss)  # <<--- GUARDAR LA PÉRDIDA DE ESTA ÉPOCA
    print(f"Época {epoch + 1}/{epochs} – Pérdida: {total_loss:.4f}")

torch.save(model.state_dict(), f"{ticker}_lstm_multivariable_model.pth")

# === PREDICCIÓN TEST ===
model.eval()
with torch.no_grad():
    pred_scaled = model(X_test).cpu().numpy()

real = scaler.inverse_transform(np.concatenate([
    y_test.cpu().numpy().reshape(-1, 1),
    np.zeros((len(y_test), input_size - 1))
], axis=1))[:, 0]

pred = scaler.inverse_transform(np.concatenate([
    pred_scaled,
    np.zeros((len(pred_scaled), input_size - 1))
], axis=1))[:, 0]

# === GUARDAR RESULTADOS ===
fechas_test = df_test["Date"].iloc[window_size:].reset_index(drop=True)
df_result = pd.DataFrame({
    "Date": fechas_test,
    "Real": real.flatten(),
    "Predicho": pred.flatten()
})
csv_path = os.path.join(SAVE_DIR, f"{ticker}_lstm_multivariable_predicciones.csv")
df_result.to_csv(csv_path, index=False)
print(f"📁 Resultados guardados en: {csv_path}")

# === MÉTRICAS ===
rmse = np.sqrt(mean_squared_error(df_result["Real"], df_result["Predicho"]))
mae = mean_absolute_error(df_result["Real"], df_result["Predicho"])
r2 = r2_score(df_result["Real"], df_result["Predicho"])
mape = mean_absolute_percentage_error(df_result["Real"], df_result["Predicho"])

print("\n📊 Métricas del Modelo Multivariable:")
print(f"   ✅ RMSE : {rmse:.4f}")
print(f"   ✅ MAE  : {mae:.4f}")
print(f"   ✅ R²   : {r2:.4f}")
print(f"   ✅ MAPE : {mape * 100:.2f}%")

# === GRAFICAR RESULTADOS (con métricas en cuadro de texto) ===
plt.figure(figsize=(12, 5))
plt.plot(df_result["Date"], df_result["Real"], label="Real", color="blue")
plt.plot(df_result["Date"], df_result["Predicho"], label="Predicción", color="orange", linestyle="--")
plt.title(f"{ticker} – LSTM Multivariable")
plt.xlabel("Fecha")
plt.ylabel("Precio Close")
plt.legend()

# Texto con métricas
textstr = f"""Métricas:
RMSE  = {rmse:.2f}
MAE   = {mae:.2f}
R²    = {r2:.2f}
MAPE  = {mape * 100:.2f}%"""

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
plot_path = os.path.join(GRAPH_DIR, f"{ticker}_lstm_multivariable_test.png")
plt.savefig(plot_path)
plt.close()
print(f"🖼️ Gráfico guardado en: {plot_path}")

# === FORECAST AUTORREGRESIVO MULTIVARIABLE A 10 DÍAS ===
forecast_horizon = 10
input_seq = test_scaled[-window_size:].copy()  # Última ventana real
preds_scaled = []

model.eval()
with torch.no_grad():
    for _ in range(forecast_horizon):
        input_tensor = torch.tensor(input_seq.reshape(1, window_size, input_size), dtype=torch.float32).to(device)
        pred_close_scaled = model(input_tensor).cpu().numpy()[0][0]
        preds_scaled.append(pred_close_scaled)

        # Nueva fila a insertar (autoregresiva)
        nueva_fila = input_seq[-1].copy()
        nueva_fila[0] = pred_close_scaled  # Solo predecimos Close, mantenemos las otras igual
        input_seq = np.vstack([input_seq[1:], nueva_fila])

# === Invertir escalado SOLO del Close ===
preds_unscaled = scaler.inverse_transform(
    np.hstack([np.array(preds_scaled).reshape(-1, 1), np.zeros((forecast_horizon, input_size - 1))])
)[:, 0]

# === Fechas futuras ===
from datetime import timedelta
last_date = df_result["Date"].iloc[-1]
future_dates = [last_date + timedelta(days=i + 1) for i in range(forecast_horizon)]

# === Intervalo de confianza ±1σ del error de test ===
std_error = np.std(df_result["Real"] - df_result["Predicho"])
lower_bound = preds_unscaled - std_error
upper_bound = preds_unscaled + std_error

# === Crear DataFrame de forecast
df_forecast = pd.DataFrame({
    "Date": future_dates,
    "Real": [np.nan] * forecast_horizon,
    "Predicho": preds_unscaled,
    "IC_lower": lower_bound,
    "IC_upper": upper_bound
})

# === Concatenar con resultados test y guardar
df_final = pd.concat([df_result, df_forecast], ignore_index=True)
df_final.to_csv(csv_path, index=False)
print(f"📁 Forecast futuro agregado al archivo: {csv_path}")

# === Graficar predicción + forecast (zoom en parte final) ===
plt.figure(figsize=(12, 5))

# Definir índice de inicio del zoom (últimos 30 días reales + 10 forecast)
n_dias_zoom = 30
fecha_inicio_zoom = df_result["Date"].iloc[-n_dias_zoom]

df_zoom = df_final[df_final["Date"] >= fecha_inicio_zoom]

plt.plot(df_zoom["Date"], df_zoom["Real"], label="Real", color="blue")
plt.plot(df_zoom["Date"], df_zoom["Predicho"], label="Predicción", color="orange", linestyle="--")
plt.fill_between(df_zoom["Date"].iloc[-forecast_horizon:],
                 df_zoom["IC_lower"].iloc[-forecast_horizon:],
                 df_zoom["IC_upper"].iloc[-forecast_horizon:],
                 color='orange', alpha=0.2, label="IC ±1σ")

plt.axvline(x=last_date, color='gray', linestyle='--', linewidth=1)
plt.title(f"{ticker} – LSTM Multivariable (Zoom últimos 30 días + Forecast)")
plt.xlabel("Fecha")
plt.ylabel("Precio Close")
plt.legend()
plt.grid(True)
plt.tight_layout()
plot_path_forecast_zoom = os.path.join(FORECAST_DIR, f"{ticker}_lstm_multivariable_forecast_10d_zoom.png")
plt.savefig(plot_path_forecast_zoom)
plt.close()
print(f"🔍 Gráfico con zoom guardado en: {plot_path_forecast_zoom}")

# === GRAFICAR FUNCIÓN DE PÉRDIDA ===
plt.figure(figsize=(10, 4))
plt.plot(range(1, epochs + 1), train_losses, label="Pérdida de entrenamiento", color="red")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.title(f"{ticker} – Función de pérdida durante el entrenamiento")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()