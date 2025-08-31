import os
import pandas as pd
from typing import List, Optional

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
    Prepara CSV para RL uniendo predicciones del LSTM con datos reales.
    Asume que los datos ya contienen todas las features necesarias y no tienen NaN.
    """
    if path_pred is None:
        path_pred = f"../3_Modelado_Precios_Futuros/results/predictions/lstm_walkforward/{ticker}_lstm_walkforward_pred.csv"
    if path_real is None:
        path_real = f"../2_Procesamiento_Datos_Stock/processed_data/{ticker}_processed.csv"
    if out_path is None:
        out_path = f"data/rl_input/{ticker}_rl_input.csv"

    # Cargar datos
    df_pred = pd.read_csv(path_pred, parse_dates=["Date"])
    df_real = pd.read_csv(path_real, parse_dates=["Date"])

    # Renombrar columna de predicción
    df_pred = df_pred.rename(columns={"Predicho": "Pred", "Pred": "Pred"})

    # Merge por fecha
    df = pd.merge(df_real, df_pred[['Date', 'Pred']], on="Date", how="inner")

    # Excluir columna Ticker si existe
    if "Ticker" in df.columns:
        df = df.drop(columns=["Ticker"])

    # Verificar que todas las columnas esperadas estén presentes
    for col in FEATURE_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing expected feature column: {col}")

    # Guardar CSV final
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    # Directorio de datos procesados
    processed_data_dir = "../2_Procesamiento_Datos_Stock/processed_data"

    # Obtener lista de archivos CSV en el directorio
    csv_files = [f for f in os.listdir(processed_data_dir) if f.endswith(".csv")]

    # Extraer tickers de los nombres de los archivos (asumiendo formato {ticker}_processed.csv)
    tickers = [f.replace("_processed.csv", "") for f in csv_files]

    # Procesar cada ticker
    for ticker in tickers:
        print(f"Preparando datos para RL con ticker: {ticker}")
        output_path = preparar_input_rl(ticker=ticker)
        print(f"Archivo CSV generado en: {output_path}")