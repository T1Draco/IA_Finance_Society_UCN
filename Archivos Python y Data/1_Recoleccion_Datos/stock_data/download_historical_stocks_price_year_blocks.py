import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import random
import logging

# === CONFIGURACIÓN GENERAL ===
TICKERS = ["AMZN", "NVDA", "AAPL", "TSLA", "MSFT", "GC=F", "ETH-USD"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "descarga.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_fecha_inicio(ticker):
    try:
        df_temp = yf.download(ticker, period="max", interval="1d", progress=False)
        return df_temp.index.min() if not df_temp.empty else None
    except Exception as e:
        logging.error(f"No se pudo obtener fecha inicial de {ticker}: {e}")
        return None

def descargar_datos_anuales(ticker, retries=3):
    archivo = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
    hoy = pd.to_datetime(datetime.today().strftime('%Y-%m-%d'))

    # Leer archivo existente
    if os.path.exists(archivo):
        try:
            df_existente = pd.read_csv(archivo, parse_dates=["Date"])
            if df_existente.empty:
                os.remove(archivo)
                df_existente = None
                start = get_fecha_inicio(ticker)
            else:
                ultima_fecha = df_existente["Date"].max()
                start = ultima_fecha + timedelta(days=1)
        except pd.errors.EmptyDataError:
            print(f"🗑️ Archivo corrupto: {archivo}. Eliminando...")
            os.remove(archivo)
            df_existente = None
            start = get_fecha_inicio(ticker)
    else:
        df_existente = None
        start = get_fecha_inicio(ticker)

    if start is None:
        print(f"⚠️ No se pudo determinar la fecha de inicio para {ticker}")
        return

    while start <= hoy:
        end = min(start + timedelta(days=364), hoy)
        print(f"📥 Descargando {ticker} desde {start.date()} hasta {end.date()}")

        for intento in range(retries):
            try:
                df = yf.download(ticker, start=start, end=end, interval="1d", progress=False, timeout=60)

                if df.empty:
                    print(f"⚠️ Datos vacíos o insuficientes ({len(df)} filas) entre {start.date()} y {end.date()}")
                    break

                # Aplanar MultiIndex si existe
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df.reset_index(inplace=True)
                df["Ticker"] = ticker

                columnas_esperadas = ["Date", "Close", "High", "Low", "Open", "Volume", "Ticker"]
                df_cols_disponibles = [col for col in columnas_esperadas if col in df.columns]

                if "Date" not in df_cols_disponibles:
                    raise ValueError(f"❌ La columna 'Date' no está en las columnas descargadas: {df.columns}")

                df = df[df_cols_disponibles]

                if df_existente is not None:
                    antes = len(df_existente)
                    df = pd.concat([df_existente, df], ignore_index=True)
                    df.drop_duplicates(subset=["Date"], keep="last", inplace=True)
                    df.sort_values("Date", inplace=True)
                    despues = len(df)
                    if despues <= antes:
                        print(f"📦 Sin nuevos datos entre {start.date()} y {end.date()}")
                        break

                df.to_csv(archivo, index=False)
                df_existente = df  # actualizar
                print(f"✅ {ticker} actualizado hasta {end.date()}")
                logging.info(f"{ticker} actualizado hasta {end.date()}")

                delay = random.uniform(10, 20)
                print(f"⏱️ Esperando {delay:.2f} segundos antes de continuar...\n")
                time.sleep(delay)
                break

            except Exception as e:
                print(f"⚠️ Error en intento {intento+1} para {ticker}: {e}")
                logging.warning(f"{ticker} ({start.date()} - {end.date()}) intento {intento+1}: {e}")
                time.sleep(10 * (intento + 1))
        else:
            logging.error(f"❌ Falló la descarga de {ticker} entre {start.date()} y {end.date()} tras {retries} intentos.")
            print(f"❌ Error persistente. Se detiene la descarga para {ticker}.")
            return
        # === AL FINAL DE descargar_datos_anuales(ticker) ===
        print(f"✅ {ticker} completamente actualizado hasta {df_existente['Date'].max().date()}")

        start = end + timedelta(days=1)

# === EJECUCIÓN PRINCIPAL ===
if __name__ == "__main__":
    for ticker in TICKERS:
        descargar_datos_anuales(ticker)

        # Mostrar última fecha descargada, aunque no se haya actualizado
        archivo = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
        if os.path.exists(archivo):
            try:
                df = pd.read_csv(archivo, parse_dates=["Date"])
                if not df.empty:
                    fecha_max = df["Date"].max().date()
                    print(f"✅ {ticker} completamente actualizado hasta {fecha_max}\n")
                    logging.info(f"{ticker} completamente actualizado hasta {fecha_max}")
                else:
                    print(f"⚠️ {ticker} no tiene datos válidos en el archivo CSV.\n")
            except Exception as e:
                print(f"❌ Error al leer {ticker}: {e}\n")

