import os
import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import random
import logging

# === CONFIGURACIÓN GENERAL ===
TICKERS = ["AAPL", "TSLA", "MSFT"]
END_DATE = datetime.today().strftime('%Y-%m-%d')

# Carpetas
RAW_DATA_DIR = os.path.join("raw_data")
LOG_DIR = os.path.join("logs")
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "descarga.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def descargar_datos(ticker, end=END_DATE, retries=3):
    archivo = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
    df_existente = None
    start = None

    if os.path.exists(archivo):
        df_existente = pd.read_csv(archivo, parse_dates=["Date"])
        ultima_fecha = df_existente["Date"].max()
        start = (ultima_fecha + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if pd.to_datetime(start) >= pd.to_datetime(end):
            print(f"📦 {ticker} ya está actualizado hasta {end}")
            logging.info(f"{ticker} ya estaba actualizado.")
            return False  # No necesita descargar

        print(f"📆 {ticker} ya tiene datos hasta {ultima_fecha.date()}, actualizando desde {start}...")
    else:
        print(f"⏳ Descargando histórico completo de {ticker} con period='max'...")

    for intento in range(retries):
        try:
            if start is None:
                df = yf.download(ticker, period="max", interval="1d", timeout=60, progress=False)
            else:
                df = yf.download(ticker, start=start, end=end, interval="1d", timeout=60, progress=False)

            if df.empty:
                print(f"⚠️ No hay datos nuevos para {ticker}")
                return False

            df.reset_index(inplace=True)
            df["Ticker"] = ticker

            if df_existente is not None:
                df = pd.concat([df_existente, df], ignore_index=True)
                df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
                df.sort_values("Date", inplace=True)

            df.to_csv(archivo, index=False)
            print(f"✅ {ticker} actualizado y guardado en {archivo}")
            logging.info(f"{ticker} actualizado correctamente.")
            return True

        except Exception as e:
            logging.warning(f"⚠️ Error al descargar {ticker} (intento {intento + 1}): {e}")
            time.sleep(10 * (intento + 1))

    logging.error(f"❌ {ticker} falló tras {retries} intentos.")
    print(f"❌ {ticker} no se pudo descargar después de {retries} intentos.")
    return False

# === EJECUCIÓN PRINCIPAL ===
if __name__ == "__main__":
    for ticker in TICKERS:
        se_descargo = descargar_datos(ticker)

        if se_descargo:
            delay = random.uniform(10, 20)
            print(f"⏱️ Esperando {delay:.2f} segundos antes del siguiente ticker...\n")
            time.sleep(delay)
