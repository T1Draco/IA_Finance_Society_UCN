import os
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import random
import logging

# === CONFIGURACIÓN GENERAL ===
TICKERS = ["AAPL", "TSLA", "MSFT"]  # puedes cambiar este subset por lotes
START_DATE = "1900-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')

# Carpetas
RAW_DATA_DIR = os.path.join("stock_data", "raw_data")
LOG_DIR = os.path.join("stock_data", "logs")
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Configurar logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "descarga.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# === FUNCIONES ===

def ya_descargado(ticker):
    return os.path.exists(os.path.join(RAW_DATA_DIR, f"{ticker}.csv"))

def descargar_datos(ticker, start=START_DATE, end=END_DATE, retries=3):
    archivo = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")

    # Si ya existe el archivo, encuentra la última fecha
    if os.path.exists(archivo):
        df_existente = pd.read_csv(archivo, parse_dates=["Date"])
        ultima_fecha = df_existente["Date"].max()
        start = (pd.to_datetime(ultima_fecha) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"📆 {ticker} ya tiene datos hasta {ultima_fecha.date()}, actualizando desde {start}...")
    else:
        df_existente = None

    if pd.to_datetime(start) >= pd.to_datetime(end):
        print(f"📦 {ticker} ya está actualizado hasta {end}")
        return

    for intento in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, interval="1d", timeout=60, progress=False)

            if df.empty:
                print(f"⚠️ No hay datos nuevos para {ticker}")
                return

            df.reset_index(inplace=True)
            df["Ticker"] = ticker

            # Combinar con datos existentes si los hay
            if df_existente is not None:
                df = pd.concat([df_existente, df], ignore_index=True)
                df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
                df.sort_values("Date", inplace=True)

            df.to_csv(archivo, index=False)
            print(f"✅ {ticker} actualizado y guardado en {archivo}")
            logging.info(f"{ticker} actualizado correctamente.")
            return

        except Exception as e:
            logging.warning(f"⚠️ Error al actualizar {ticker} (intento {intento + 1}): {e}")
            wait_time = 10 * (intento + 1)
            time.sleep(wait_time)

    logging.error(f"❌ {ticker} falló tras {retries} intentos.")
    print(f"❌ {ticker} no se pudo actualizar después de {retries} intentos.")

# === EJECUCIÓN PRINCIPAL ===

if __name__ == "__main__":
    for ticker in TICKERS:
        if ya_descargado(ticker):
            print(f"📦 {ticker} ya descargado. Se omite.")
            continue

        descargar_datos(ticker)
        delay = random.uniform(10, 20)
        print(f"⏱️ Esperando {delay:.2f} segundos antes del siguiente ticker...\n")
        time.sleep(delay)
