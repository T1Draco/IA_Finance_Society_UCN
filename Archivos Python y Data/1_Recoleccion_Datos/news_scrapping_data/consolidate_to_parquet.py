import os
import pandas as pd

def consolidar_json_a_parquet(ticker, base_dir="diarios_json", output_dir="consolidados_parquet"):
    fuentes = ["finviz", "googlenews", "reddit", "yahoofinance"]
    frames = []

    base_path = os.path.join("1_Recoleccion_Datos", "news_scrapping_data", base_dir)
    output_path = os.path.join("1_Recoleccion_Datos", "news_scrapping_data", output_dir)
    os.makedirs(output_path, exist_ok=True)

    for fuente in fuentes:
        fuente_dir = os.path.join(base_path, ticker, fuente)
        if not os.path.exists(fuente_dir):
            continue
        for archivo in os.listdir(fuente_dir):
            if archivo.endswith(".json"):
                path = os.path.join(fuente_dir, archivo)
                try:
                    df = pd.read_json(path)
                    frames.append(df)
                except Exception as e:
                    print(f"[⚠] Error leyendo {path}: {e}")

    if frames:
        df_total = pd.concat(frames, ignore_index=True)
        parquet_path = os.path.join(output_path, f"{ticker}_news.parquet")
        df_total.to_parquet(parquet_path, index=False)
        print(f"[✔] Consolidado: {parquet_path}")
    else:
        print(f"[⚠] No hay noticias para {ticker} en {base_dir}")

# === Uso ejemplo ===
if __name__ == "__main__":
    for ticker in ["AAPL", "TSLA", "MSFT", "NVDA", "AMZN"]:
        consolidar_json_a_parquet(ticker)
