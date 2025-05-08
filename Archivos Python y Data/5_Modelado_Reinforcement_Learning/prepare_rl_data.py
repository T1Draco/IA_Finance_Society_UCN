import pandas as pd
import os


def preparar_input_rl(ticker='AAPL'):
    path_pred = f"../3_Modelado_Precios_Futuros/results/predictions/lstm/{ticker}_lstm_pytorch_predicciones.csv"
    path_real = f"../2_Procesamiento_Datos_Stock/processed_data/{ticker}_processed.csv"

    df_pred = pd.read_csv(path_pred, parse_dates=["Date"])
    df_real = pd.read_csv(path_real, parse_dates=["Date"])

    # Renombramos columna de predicción
    df_pred = df_pred.rename(columns={"Predicho": "Pred"})

    # Unimos por fecha manteniendo todos los indicadores
    df = pd.merge(df_real, df_pred[['Date', 'Pred']], on="Date", how="inner")

    # Reordenamos columnas, pero mantenemos todos los indicadores
    columnas_prioritarias = ['Date', 'Close', 'Pred']
    columnas_extra = [col for col in df.columns if col not in columnas_prioritarias]
    df = df[columnas_prioritarias + columnas_extra]
    df["sma_diff"] = df["SMA_20"] - df["SMA_50"]

    # Guardar archivo final
    os.makedirs("./data/rl_input", exist_ok=True)
    output_path = f"./data/rl_input/{ticker}_rl_input.csv"
    df.to_csv(output_path, index=False)
    print(f"[✔] Archivo guardado: {output_path}")


if __name__ == "__main__":
    for ticker in ["AAPL", "AMZN", "MSFT", "NVDA", "TSLA"]:
        preparar_input_rl(ticker)
