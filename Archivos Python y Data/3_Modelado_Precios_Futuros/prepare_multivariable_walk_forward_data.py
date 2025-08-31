import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "..", "2_Procesamiento_Datos_Stock", "processed_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "multivariable_walkforward_hibrid_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VENTANA = 60
HORIZONTE = 1
FEATURES = ["Close", "High", "Low", "Volume", "retorno_simple", "retorno_log", "SMA_20", "SMA_50", "RSI_14", "MACD", "MACD_signal"]


def crear_secuencias_hibridas(df, scaler, ventana=VENTANA, horizonte=HORIZONTE):
    X_list, y_list, fechas_list = [], [], []

    for t in range(len(df) - ventana - horizonte + 1):
        window_df = df.iloc[t:t + ventana][FEATURES].copy()
        target = df.iloc[t + ventana + horizonte - 1]["Close"]
        target_fecha = df.iloc[t + ventana + horizonte - 1]["Date"]

        # Escalar usando el scaler global
        window_scaled = scaler.transform(window_df)

        X_list.append(window_scaled)
        y_list.append(target)
        fechas_list.append(target_fecha)

    return np.array(X_list), np.array(y_list), np.array(fechas_list)


# === Procesar todos los tickers ===
for archivo in os.listdir(PROCESSED_DATA_DIR):
    if archivo.endswith("_processed.csv"):
        ticker = archivo.replace("_processed.csv", "")
        print(f"\n🔄 Procesando {ticker} con walk-forward híbrido...")

        df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, archivo), parse_dates=["Date"])
        df = df.sort_values("Date").dropna(subset=FEATURES)

        # === Escalar con un scaler global basado en todo el histórico (puedes elegir solo un bloque inicial) ===
        scaler = MinMaxScaler()
        scaler.fit(df[FEATURES])  # Ajuste global
        joblib.dump(scaler, os.path.join(OUTPUT_DIR, f"{ticker}_global_scaler.pkl"))

        # === Crear secuencias con este scaler ===
        X, y, fechas = crear_secuencias_hibridas(df, scaler, VENTANA, HORIZONTE)

        # Guardar arrays
        np.save(os.path.join(OUTPUT_DIR, f"{ticker}_X.npy"), X)
        np.save(os.path.join(OUTPUT_DIR, f"{ticker}_y.npy"), y)
        np.save(os.path.join(OUTPUT_DIR, f"{ticker}_fechas.npy"), fechas)

        print(f"✅ {ticker} listo → {X.shape[0]} secuencias de {VENTANA} pasos")

