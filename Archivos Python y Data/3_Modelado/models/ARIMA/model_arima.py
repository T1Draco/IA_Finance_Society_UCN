import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

# === CONFIGURACIÓN ===
ticker = "TSLA"
train_path = f"../../data/train_test_split/{ticker}_train.csv"
test_path = f"../../data/train_test_split/{ticker}_test.csv"
SAVE_DIR = "../../results/predictions"
GRAPH_DIR = "../../results/plots"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

# === CARGA DE DATOS ===
df_train = pd.read_csv(train_path, parse_dates=["Date"])
df_test = pd.read_csv(test_path, parse_dates=["Date"])
train_series = df_train["Close"]
test_series = df_test["Close"]

# === TEST DE ESTACIONARIEDAD ===
result_adf = adfuller(train_series)
print(f"📉 ADF p-value: {result_adf[1]:.4f}")
if result_adf[1] > 0.05:
    print("⚠️ Serie no estacionaria, considera aplicar diferenciación.")

# === ENTRENAMIENTO AUTOMÁTICO ARIMA ===
print("🔍 Ejecutando auto_arima...")

auto_model = auto_arima(train_series,
                        seasonal=False,
                        stepwise=True,
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        max_p=5, max_q=5, max_d=2)

print(f"✅ Modelo óptimo encontrado: {auto_model.order}")
print(auto_model.summary())

# === ENTRENAMIENTO MANUAL FORZADO ===
p, d, q = auto_model.order
manual_model = ARIMA(train_series, order=(p, d, q))
manual_fit = manual_model.fit()

# === PREDICCIÓN ===
n_periods = len(df_test)
forecast = manual_fit.forecast(steps=n_periods)

# Convertir a numpy si es una serie con índice
if isinstance(forecast, pd.Series):
    forecast = forecast.values

# Verificación
print(f"📈 Forecast generado (primeros 5): {forecast[:5]}")
print(f"🔢 Largo del forecast: {len(forecast)}")
print(f"🔍 Longitudes → Date: {len(df_test['Date'])}, Real: {len(test_series)}, Predicho: {len(forecast)}")

# === Asegurar longitud común ===
min_len = min(len(df_test), len(test_series), len(forecast))

# Crear dataframe de resultados
df_result = pd.DataFrame({
    "Date": df_test["Date"].iloc[:min_len].values,
    "Real": test_series.iloc[:min_len].values,
    "Predicho": forecast[:min_len]
})

# Guardado seguro
csv_path = os.path.join(SAVE_DIR, f"{ticker}_arima_predicciones.csv")
print(f"💾 Guardando predicciones en: {csv_path}")
df_result.to_csv(csv_path, index=False)

# Eliminar posibles NaNs
df_result.dropna(inplace=True)

# === EVALUACIÓN SOLO SI HAY DATOS PREDICHOS ===
if len(df_result) > 0:
    rmse = np.sqrt(mean_squared_error(df_result["Real"], df_result["Predicho"]))
    mae = mean_absolute_error(df_result["Real"], df_result["Predicho"])
    print(f"📊 RMSE: {rmse:.2f} | MAE: {mae:.2f}")
else:
    print("❌ No hay predicciones válidas para evaluar.")

# === GRAFICAR ===
plt.figure(figsize=(12, 5))
plt.plot(df_train["Date"], df_train["Close"], label="Train", color="gray", alpha=0.6)
plt.plot(df_result["Date"], df_result["Real"], label="Test Real", color="blue")
plt.plot(df_result["Date"], df_result["Predicho"], label="Predicción ARIMA", color="red", linestyle="--")
plt.title(f"{ticker} – Predicción ARIMA {auto_model.order}")
plt.xlabel("Fecha")
plt.ylabel("Precio Close")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, f"{ticker}_arima_forecast.png"))
plt.close()
print(f"🖼️ Gráfico guardado: {ticker}_arima_forecast.png")
