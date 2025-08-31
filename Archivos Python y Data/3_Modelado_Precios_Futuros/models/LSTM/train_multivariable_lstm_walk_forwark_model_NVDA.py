import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from datetime import timedelta

# ================= CONFIGURACIÓN =================
ticker = "NVDA"
window_size = 60
epochs = 60
batch_size = 32
forecast_horizon = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 5e-4  # learning rate más conservador

# ================= RUTAS =================
DATA_DIR = "../../data/multivariable_walkforward_hibrid_data"
SAVE_DIR = "../../results/predictions/lstm_walkforward"
GRAPH_DIR = "../../results/plots/lstm_walkforward"
FORECAST_DIR = os.path.join(GRAPH_DIR, f"forecast")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(FORECAST_DIR, exist_ok=True)

# ================= CARGAR DATOS =================
X_all = np.load(os.path.join(DATA_DIR, f"{ticker}_X.npy")).astype(np.float32)
y_all = np.load(os.path.join(DATA_DIR, f"{ticker}_y.npy")).astype(np.float32)
fechas_all = np.load(os.path.join(DATA_DIR, f"{ticker}_fechas.npy"), allow_pickle=True)

# Escalar con StandardScaler global
scaler = StandardScaler()
nsamples, nsteps, nfeatures = X_all.shape
X_flat = X_all.reshape(-1, nfeatures)
X_scaled_flat = scaler.fit_transform(X_flat)
X_all_scaled = X_scaled_flat.reshape(nsamples, nsteps, nfeatures)

y_scaled = scaler.transform(np.hstack([y_all.reshape(-1,1), np.zeros((len(y_all), nfeatures-1))]))[:,0]

# Convertir a tensores
X_tensor = torch.tensor(X_all_scaled, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

# ================= MODELO =================
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel(nfeatures).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# ================= ENTRENAMIENTO =================
train_losses = []
model.train()
for epoch in range(epochs):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch).squeeze()
        loss = criterion(output, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss)
    print(f"Época {epoch+1}/{epochs} – Pérdida: {total_loss:.4f}")

# Guardar modelo
save_dir = "./models_pytorch"
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_dir, f"{ticker}_lstm_walkforward.pth"))
joblib.dump(scaler, os.path.join(save_dir, f"{ticker}_walkforward_scaler.pkl"))

# ================= PREDICCIÓN WALK-FORWARD =================
model.eval()
with torch.no_grad():
    pred_scaled = model(X_tensor).cpu().numpy()

# Invertir escala solo Close
real = y_all
pred = scaler.inverse_transform(np.hstack([pred_scaled.reshape(-1,1), np.zeros((len(pred_scaled), nfeatures-1))]))[:,0]

df_result = pd.DataFrame({
    "Date": fechas_all,
    "Real": real,
    "Predicho": pred
})

# ================= FORECAST AUTORREGRESIVO =================
input_seq = X_all_scaled[-1].copy()
preds_scaled = []

model.eval()
with torch.no_grad():
    for _ in range(forecast_horizon):
        input_tensor = torch.tensor(input_seq.reshape(1, window_size, nfeatures), dtype=torch.float32).to(device)
        pred_close_scaled = model(input_tensor).cpu().numpy()[0][0]
        preds_scaled.append(pred_close_scaled)
        nueva_fila = input_seq[-1].copy()
        nueva_fila[0] = pred_close_scaled
        input_seq = np.vstack([input_seq[1:], nueva_fila])

preds_unscaled = scaler.inverse_transform(np.hstack([np.array(preds_scaled).reshape(-1,1),
                                                     np.zeros((forecast_horizon, nfeatures-1))]))[:,0]

last_date = pd.to_datetime(df_result["Date"].iloc[-1])
future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_horizon)]

# Intervalo de confianza ±1σ
std_error = np.std(df_result["Real"] - df_result["Predicho"])
lower_bound = preds_unscaled - std_error
upper_bound = preds_unscaled + std_error

df_forecast = pd.DataFrame({
    "Date": future_dates,
    "Real": [np.nan]*forecast_horizon,
    "Predicho": preds_unscaled,
    "IC_lower": lower_bound,
    "IC_upper": upper_bound
})

df_final = pd.concat([df_result, df_forecast], ignore_index=True)
df_final.to_csv(os.path.join(SAVE_DIR, f"{ticker}_lstm_walkforward_pred.csv"), index=False)

# ================= GRÁFICOS =================
# 1️⃣ Predicción vs Real completa con cuadro de métricas
plt.figure(figsize=(12,5))
plt.plot(df_final["Date"], df_final["Real"], label="Real", color="blue")
plt.plot(df_final["Date"], df_final["Predicho"], label="Predicción", color="orange", linestyle="--")
plt.fill_between(df_final["Date"].iloc[-forecast_horizon:], lower_bound, upper_bound,
                 color='orange', alpha=0.2, label="IC ±1σ")

rmse = np.sqrt(mean_squared_error(df_result["Real"], df_result["Predicho"]))
mae = mean_absolute_error(df_result["Real"], df_result["Predicho"])
r2 = r2_score(df_result["Real"], df_result["Predicho"])
mape = mean_absolute_percentage_error(df_result["Real"], df_result["Predicho"])
# Cuadro de métricas
metrics_text = (
    f"RMSE: {rmse:.4f}\n"
    f"MAE: {mae:.4f}\n"
    f"R²: {r2:.4f}\n"
    f"MAPE: {mape*100:.2f}%"
)
plt.gca().text(0.02, 0.75, metrics_text, transform=plt.gca().transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

plt.title(f"{ticker} – Walk-forward LSTM")
plt.xlabel("Fecha")
plt.ylabel("Precio Close")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, f"{ticker}_walkforward_full.png"))
plt.close()

# 2️⃣ Zoom últimos 30 días + forecast
n_dias_zoom = 30
fecha_inicio_zoom = df_result["Date"].iloc[-n_dias_zoom]
df_zoom = df_final[df_final["Date"] >= fecha_inicio_zoom]
plt.figure(figsize=(12,5))
plt.plot(df_zoom["Date"], df_zoom["Real"], label="Real", color="blue")
plt.plot(df_zoom["Date"], df_zoom["Predicho"], label="Predicción", color="orange", linestyle="--")
plt.fill_between(df_zoom["Date"].iloc[-forecast_horizon:], lower_bound, upper_bound,
                 color='orange', alpha=0.2, label="IC ±1σ")
plt.axvline(x=last_date, color='gray', linestyle='--', linewidth=1)
plt.title(f"{ticker} – Zoom últimos 30 días + Forecast")
plt.xlabel("Fecha")
plt.ylabel("Precio Close")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(FORECAST_DIR, f"{ticker}_walkforward_zoom.png"))
plt.close()

# 3️⃣ Error a lo largo del tiempo
plt.figure(figsize=(12,4))
plt.plot(df_result["Date"], df_result["Predicho"] - df_result["Real"], color="red")
plt.title(f"{ticker} – Error Walk-forward (Predicho - Real)")
plt.xlabel("Fecha")
plt.ylabel("Error")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, f"{ticker}_walkforward_error.png"))
plt.close()

# 4️⃣ Función de pérdida
plt.figure(figsize=(10,4))
plt.plot(range(1, epochs+1), train_losses, label="Pérdida entrenamiento", color="red")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.title(f"{ticker} – Función de pérdida durante entrenamiento")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, f"{ticker}_walkforward_loss.png"))
plt.close()

# ================= MÉTRICAS =================
print("\n📊 Métricas Walk-forward:")
print(f"   ✅ RMSE  : {rmse:.4f}")
print(f"   ✅ MAE   : {mae:.4f}")
print(f"   ✅ R²    : {r2:.4f}")
print(f"   ✅ MAPE  : {mape*100:.2f}%")
