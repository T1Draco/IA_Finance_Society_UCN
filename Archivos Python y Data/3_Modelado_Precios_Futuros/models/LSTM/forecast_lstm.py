import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from datetime import timedelta

# Config
ticker = "AAPL"
window_size = 60
forecast_horizon = 10  # días a futuro
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rutas
SCALED_DIR = "../../data/scaled"
MODEL_PATH = "base_lstm_model.pth"
SAVE_PATH = f"../../results/predictions/lstm_forecast/{ticker}_forecast.csv"
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# === Modelo ===
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Cargar modelo y scaler
model = LSTMModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
scaler = joblib.load(os.path.join(SCALED_DIR, f"{ticker}_scaler.pkl"))

# Últimos datos reales escalados
df_hist = pd.read_csv(f"../../../1_Recoleccion_Datos/stock_data/raw_data/{ticker}.csv", parse_dates=["Date"])
last_data = df_hist["Close"].values[-window_size:]
scaled_last = scaler.transform(last_data.reshape(-1, 1))

# === Forecast ===
predictions = []
dates = []
input_seq = torch.tensor(scaled_last.reshape(1, -1, 1), dtype=torch.float32).to(device)
last_date = df_hist["Date"].iloc[-1]

with torch.no_grad():
    for i in range(forecast_horizon):
        pred = model(input_seq).cpu().numpy()
        predictions.append(pred[0][0])
        last_date += timedelta(days=1)
        dates.append(last_date)

        # Update input sequence
        new_input = np.append(input_seq.cpu().numpy().squeeze()[1:], pred)
        input_seq = torch.tensor(new_input.reshape(1, window_size, 1), dtype=torch.float32).to(device)

# Inversión del escalado
predictions_inv = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Guardar
df_forecast = pd.DataFrame({
    "Date": dates,
    "Forecast": predictions_inv
})
df_forecast.to_csv(SAVE_PATH, index=False)
print(f"📈 Forecast generado y guardado en {SAVE_PATH}")
