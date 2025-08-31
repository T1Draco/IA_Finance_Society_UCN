# env/data_prep_lstm.py
import os
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.preprocessing import StandardScaler

# ==========================================
FEATURE_COLS = [
    "Close", "High", "Low", "Volume",
    "retorno_simple", "retorno_log",
    "SMA_20", "SMA_50",
    "RSI_14", "MACD", "MACD_signal"
]

def preparar_input_rl(ticker='GOLD',
                       path_pred=None,
                       path_real=None,
                       out_path=None):
    """
    Prepara CSV para RL uniendo predicciones del LSTM con datos reales,
    calculando SMA y retornos, y asegurando que no haya NaN iniciales.
    """
    if path_pred is None:
        path_pred = f"../3_Modelado_Precios_Futuros/results/predictions/lstm_multivariable/{ticker}_lstm_multivariable_predicciones.csv"
    if path_real is None:
        path_real = f"../2_Procesamiento_Datos_Stock/processed_data/{ticker}_processed.csv"
    if out_path is None:
        out_path = f"./data/rl_input/{ticker}_rl_input.csv"

    df_pred = pd.read_csv(path_pred, parse_dates=["Date"])
    df_real = pd.read_csv(path_real, parse_dates=["Date"])

    # Renombrar columna de predicción
    df_pred = df_pred.rename(columns={"Predicho": "Pred", "Pred": "Pred"})

    # Merge por fecha
    df = pd.merge(df_real, df_pred[['Date', 'Pred']], on="Date", how="inner")

    # Excluir columna Ticker si existe
    if "Ticker" in df.columns:
        df = df.drop(columns=["Ticker"])

    # -----------------------
    # Feature engineering
    # -----------------------
    if "Close" not in df.columns:
        raise ValueError("No Close column found, required for SMA and returns.")

    # Medias móviles
    df["SMA_20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["SMA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    # Reemplazar posibles NaN iniciales
    df["SMA_20"].fillna(method="bfill", inplace=True)
    df["SMA_50"].fillna(method="bfill", inplace=True)
    df["sma_diff"] = df["SMA_20"] - df["SMA_50"]

    # Retornos
    df["retorno_simple"] = df["Close"].pct_change().fillna(0)
    df["retorno_log"] = np.log(df["Close"].pct_change().add(1).fillna(1))

    # Rellenar cualquier NaN residual en features
    feature_cols = FEATURE_COLS
    for col in feature_cols:
        if col in df.columns:
            df[col].fillna(method="bfill", inplace=True)
            df[col].fillna(method="ffill", inplace=True)
        else:
            raise ValueError(f"Missing expected feature column: {col}")

    # Guardar CSV final
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
