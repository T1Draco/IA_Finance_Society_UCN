import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "..", "2_Procesamiento_Datos_Stock", "processed_data")
SPLIT_DIR = os.path.join(BASE_DIR, "data", "train_test_split")
SCALED_DIR = os.path.join(BASE_DIR, "data", "scaled")
os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(SCALED_DIR, exist_ok=True)

# Porcentaje de entrenamiento
split_ratio = 0.8

for archivo in os.listdir(PROCESSED_DATA_DIR):
    if archivo.endswith("_processed.csv"):
        ticker = archivo.replace("_processed.csv", "")
        df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, archivo), parse_dates=["Date"])
        df.sort_values("Date", inplace=True)

        # Usamos solo la columna Close
        series = df[["Date", "Close"]].dropna()

        # Configuración
        AÑOS_TEST = 2  # Cambia esto a 1 si deseas usar solo el último año

        # Fecha de corte dinámica por ticker
        fecha_corte = series["Date"].max() - pd.DateOffset(years=AÑOS_TEST)

        # Split temporal basado en fecha
        df_train = series[series["Date"] < fecha_corte]
        df_test = series[series["Date"] >= fecha_corte]

        # Guardar splits
        df_train.to_csv(os.path.join(SPLIT_DIR, f"{ticker}_train.csv"), index=False)
        df_test.to_csv(os.path.join(SPLIT_DIR, f"{ticker}_test.csv"), index=False)

        # Escalar solo para LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(df_train[["Close"]])
        scaled_test = scaler.transform(df_test[["Close"]])

        np.save(os.path.join(SCALED_DIR, f"{ticker}_train_scaled.npy"), scaled_train)
        np.save(os.path.join(SCALED_DIR, f"{ticker}_test_scaled.npy"), scaled_test)
        joblib.dump(scaler, os.path.join(SCALED_DIR, f"{ticker}_scaler.pkl"))

        '''
        .npy: formato binario rápido y ligero para arrays NumPy → ideal para LSTM (X, y)

        .pkl: el scaler original para:

        Revertir predicciones (inverse_transform)

        Escalar futuros datos reales de entrada (consistencia)
        '''

        print(f"✅ {ticker}: split y escalado guardados.")

        import matplotlib.pyplot as plt

        # Crear carpeta para los gráficos
        PLOT_DIR = os.path.join(BASE_DIR, "graficos_split")
        os.makedirs(PLOT_DIR, exist_ok=True)

        # Crear gráfico visual del split con línea de corte
        plt.figure(figsize=(12, 5))
        plt.plot(df_train["Date"], df_train["Close"], label="Train", color="blue")
        plt.plot(df_test["Date"], df_test["Close"], label="Test", color="red")

        # Línea vertical en la fecha de corte
        plt.axvline(x=fecha_corte, color='black', linestyle='--', linewidth=1, label='Fecha de corte')

        # Anotación de la fecha exacta
        plt.text(fecha_corte, df["Close"].max() * 0.93,
                 fecha_corte.strftime('%Y-%m-%d'),
                 rotation=90, color='black', verticalalignment='center')

        plt.title(f"{ticker} – Train/Test Split ({AÑOS_TEST} años)")
        plt.xlabel("Fecha")
        plt.ylabel("Precio Close")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{ticker}_split.png"))
        plt.close()
        print(f"Gráfico guardado: {ticker}_split.png")


