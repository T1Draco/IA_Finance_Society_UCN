import os
import pandas as pd
import numpy as np
from technical_indicators import calcular_sma, calcular_rsi, calcular_macd

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "..", "1_Recoleccion_Datos", "stock_data", "raw_data")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


# Procesamiento por archivo
for archivo in os.listdir(RAW_DATA_DIR):
    if archivo.endswith(".csv"):
        ticker = archivo.replace(".csv", "")
        ruta_archivo = os.path.join(RAW_DATA_DIR, archivo)
        print(f"🔍 Procesando {ticker}...")

        # Cargar el archivo
        df = pd.read_csv(ruta_archivo, parse_dates=["Date"])

        # Eliminar fila con tickers repetidos si está presente
        if df.iloc[0].isnull().all() and df.iloc[1].str.contains(ticker).all():
            print("⚠️ Fila con tickers repetidos detectada. Eliminando fila 1...")
            df = df.drop(index=1).reset_index(drop=True)

        # Convertir columnas numéricas a float (por si vienen como texto)
        columnas_numericas = ["Close", "Open", "High", "Low", "Volume"]
        df[columnas_numericas] = df[columnas_numericas].apply(pd.to_numeric, errors='coerce')
        print("Primer valor:", df["Close"].iloc[0])
        print("Último valor:", df["Close"].iloc[-1])
        df.sort_values("Date", inplace=True)

        # Retornos
        df["retorno_simple"] = df["Close"].pct_change()
        df["retorno_log"] = (df["Close"] / df["Close"].shift(1)).apply(lambda x: np.log(x) if x > 0 else 0)

        # Indicadores técnicos
        df["SMA_20"] = calcular_sma(df, ventana=20)
        df["SMA_50"] = calcular_sma(df, ventana=50)
        df["RSI_14"] = calcular_rsi(df, ventana=14)
        df["MACD"], df["MACD_signal"] = calcular_macd(df)

        # Eliminar la primera fila con NaN (causado por pct_change, RSI, etc.)
        df.dropna(subset=["Close", "retorno_simple"], inplace=True)

        # Guardar archivo procesado
        ruta_salida = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_processed.csv")
        df.to_csv(ruta_salida, index=False)
        print(f"✅ Procesado {ticker} guardado en: {ruta_salida}")
