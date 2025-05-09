import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# === Configuración ===
split_ratio = 0.8  # 80% train

# === Rutas base ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "..", "2_Procesamiento_Datos_Stock", "processed_data")
SPLIT_DIR = os.path.join(BASE_DIR, "data", "train_test_split")
SCALED_DIR = os.path.join(BASE_DIR, "data/multivariable_data", "multivariable_scaled")
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(SCALED_DIR, exist_ok=True)

# === Procesar todos los archivos ===
for archivo in os.listdir(PROCESSED_DATA_DIR):
    if archivo.endswith(".csv"):
        ticker = archivo.replace("_processed.csv", "")
        print(f"\n🔄 Procesando {ticker}...")

        # === Cargar archivo ===
        df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, archivo), parse_dates=["Date"])
        df = df.dropna()

        # === Selección de features ===
        features = ["Close", "Volume", "retorno_log", "SMA_20", "SMA_50", "RSI_14", "MACD", "MACD_signal"]
        df_features = df[features]

        # === División temporal ===
        n_train = int(len(df_features) * split_ratio)
        train = df_features.iloc[:n_train]
        test = df_features.iloc[n_train:]

        # === Escalar ===
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train)
        test_scaled = scaler.transform(test)

        # === Guardar escalados ===
        np.save(os.path.join(SCALED_DIR, f"{ticker}_train_scaled.npy"), train_scaled)
        np.save(os.path.join(SCALED_DIR, f"{ticker}_test_scaled.npy"), test_scaled)
        joblib.dump(scaler, os.path.join(SCALED_DIR, f"{ticker}_scaler.pkl"))

        # === Guardar datos crudos divididos ===
        df.iloc[:n_train].to_csv(os.path.join(SPLIT_DIR, f"{ticker}_train.csv"), index=False)
        df.iloc[n_train:].to_csv(os.path.join(SPLIT_DIR, f"{ticker}_test.csv"), index=False)

        print(f"✅ {ticker} procesado correctamente.")
