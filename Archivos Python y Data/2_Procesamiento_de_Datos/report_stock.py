import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def calcular_drawdown(series):
    max_acumulado = series.cummax()
    drawdown = (series - max_acumulado) / max_acumulado
    return drawdown.min() * 100  # en %

for archivo in os.listdir(PROCESSED_DATA_DIR):
    if archivo.endswith("_processed.csv"):
        ticker = archivo.replace("_processed.csv", "")
        print(f" Generando reporte para {ticker}...")

        df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, archivo), parse_dates=["Date"])
        df.sort_values("Date", inplace=True)

        close = df["Close"]
        retorno_total = (close.iloc[-1] / close.iloc[0] - 1) * 100
        retorno_medio = df["retorno_simple"].mean() * 100
        volatilidad = df["retorno_simple"].std() * 100
        drawdown = calcular_drawdown(close)

        reporte = {
            "Ticker": ticker,
            "Fecha inicial": df["Date"].min().date(),
            "Fecha final": df["Date"].max().date(),
            "Retorno total (%)": retorno_total,
            "Retorno medio diario (%)": retorno_medio,
            "Volatilidad diaria (%)": volatilidad,
            "Precio máximo": close.max(),
            "Precio mínimo": close.min(),
            "Drawdown máximo (%)": drawdown,
            "Volumen promedio": df["Volume"].mean(),
            "Volumen máximo": df["Volume"].max(),
            "RSI promedio": df["RSI_14"].mean() if "RSI_14" in df else np.nan,
            "RSI máximo": df["RSI_14"].max() if "RSI_14" in df else np.nan,
            "RSI mínimo": df["RSI_14"].min() if "RSI_14" in df else np.nan
        }

        # Guardar como CSV individual por ticker
        df_reporte = pd.DataFrame([reporte])
        df_reporte.to_csv(os.path.join(REPORTS_DIR, f"{ticker}_reporte.csv"), index=False)
        print(f" Reporte guardado: reportes/{ticker}_reporte.csv")
