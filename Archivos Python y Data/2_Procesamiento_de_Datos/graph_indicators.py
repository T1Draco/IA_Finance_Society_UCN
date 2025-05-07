import os
import pandas as pd
import matplotlib.pyplot as plt

# Directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Archivos procesados
for archivo in os.listdir(PROCESSED_DATA_DIR):
    if archivo.endswith("_processed.csv"):
        ticker = archivo.replace("_processed.csv", "")
        print(f"📈 Graficando {ticker}...")

        df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, archivo), parse_dates=["Date"])
        df.set_index("Date", inplace=True)

        fig, axs = plt.subplots(5, 1, figsize=(15, 12), sharex=True,
                                gridspec_kw={"height_ratios": [3, 1, 1, 1, 1]})
        fig.suptitle(f"{ticker} – Histórico + Indicadores Técnicos", fontsize=16)

        # Precio de cierre + SMA
        axs[0].plot(df.index, df["Close"], label="Close", linewidth=1)
        if "SMA_20" in df:
            axs[0].plot(df.index, df["SMA_20"], label="SMA 20", linestyle="--")
        axs[0].set_ylabel("Precio")
        axs[0].legend()
        axs[0].grid(True)

        # Volumen
        axs[1].bar(df.index, df["Volume"], label="Volume", color="gray", alpha=0.6)
        axs[1].set_ylabel("Volumen")
        axs[1].legend()
        axs[1].grid(True)

        # Retorno simple
        if "retorno_simple" in df:
            axs[2].plot(df.index, df["retorno_simple"], label="Retorno simple", color="brown")
            axs[2].axhline(0, color="black", linewidth=0.5)
            axs[2].set_ylabel("Retorno")
            axs[2].legend()
            axs[2].grid(True)

        # RSI
        if "RSI_14" in df:
            axs[3].plot(df.index, df["RSI_14"], label="RSI 14", color="orange")
            axs[3].axhline(70, color="red", linestyle="--", linewidth=0.8)
            axs[3].axhline(30, color="green", linestyle="--", linewidth=0.8)
            axs[3].set_ylabel("RSI")
            axs[3].legend()
            axs[3].grid(True)

        # MACD
        if "MACD" in df and "MACD_signal" in df:
            axs[4].plot(df.index, df["MACD"], label="MACD", color="blue")
            axs[4].plot(df.index, df["MACD_signal"], label="Signal", color="magenta", linestyle="--")
            axs[4].axhline(0, color="black", linewidth=0.5)
            axs[4].set_ylabel("MACD")
            axs[4].legend()
            axs[4].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(PLOTS_DIR, f"{ticker}_indicadores.png"))
        plt.close()
        print(f"✅ Gráfico guardado: plots/{ticker}_indicadores.png")
