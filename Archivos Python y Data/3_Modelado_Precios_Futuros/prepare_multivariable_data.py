import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# === Configuración ===
AÑOS_TEST = 2  # último N años para test

# === Rutas base ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "..", "2_Procesamiento_Datos_Stock", "processed_data")
SPLIT_DIR = os.path.join(BASE_DIR, "data", "train_test_split")
SCALED_DIR = os.path.join(BASE_DIR, "data", "multivariable_data", "multivariable_scaled")
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(SCALED_DIR, exist_ok=True)

# === Procesar todos los archivos ===
for archivo in os.listdir(PROCESSED_DATA_DIR):
    if archivo.endswith("_processed.csv"):
        ticker = archivo.replace("_processed.csv", "")
        print(f"\n🔄 Procesando {ticker}...")

        # === Cargar archivo ===
        df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, archivo), parse_dates=["Date"])
        df = df.sort_values("Date").dropna()

        # === Selección de features ===
        features = ["Close", "Volume", "retorno_log", "SMA_20", "SMA_50", "RSI_14", "MACD", "MACD_signal"]
        df = df[["Date"] + features].dropna()

        # === División temporal por fecha ===
        fecha_corte = df["Date"].max() - pd.DateOffset(years=AÑOS_TEST)
        df_train = df[df["Date"] < fecha_corte]
        df_test = df[df["Date"] >= fecha_corte]

        # === Escalar ===
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(df_train[features])
        test_scaled = scaler.transform(df_test[features])

        # === Guardar escalados ===
        np.save(os.path.join(SCALED_DIR, f"{ticker}_train_scaled.npy"), train_scaled)
        np.save(os.path.join(SCALED_DIR, f"{ticker}_test_scaled.npy"), test_scaled)
        joblib.dump(scaler, os.path.join(SCALED_DIR, f"{ticker}_scaler.pkl"))

        # === Guardar datos crudos divididos ===
        df_train.to_csv(os.path.join(SPLIT_DIR, f"{ticker}_train.csv"), index=False)
        df_test.to_csv(os.path.join(SPLIT_DIR, f"{ticker}_test.csv"), index=False)

        print(f"✅ {ticker} procesado correctamente con corte en {fecha_corte.date()}")
