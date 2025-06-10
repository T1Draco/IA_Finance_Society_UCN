import pandas as pd
import os

folder = "processed_data"
output_folder = "excel_para_sharepoint"
os.makedirs(output_folder, exist_ok=True)

# columnas para cada tabla
cols_precio = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
cols_indicadores = ['Date', 'Ticker', 'retorno_simple', 'retorno_log', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_signal']

# Diccionarios para acumular DataFrames por hoja
precio_dfs = {}
indicador_dfs = {}

for file in os.listdir(folder):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(folder, file))
        df['Date'] = pd.to_datetime(df['Date'])

        # Filtrar últimos 10 años (aquí solo desde 2024 como ejemplo)
        df = df[df['Date'].dt.year >= 2024]

        # Extraer ticker
        ticker = file.split("_")[0]
        df['Ticker'] = ticker

        # Mover Ticker a la segunda columna
        cols = df.columns.tolist()
        if 'Ticker' in cols:
            cols.remove('Ticker')
            cols.insert(1, 'Ticker')
            df = df[cols]

        # Separar y guardar
        df_precio = df[[c for c in cols_precio if c in df.columns]].copy()
        df_indicador = df[[c for c in cols_indicadores if c in df.columns]].copy()

        # Formatear fecha como mm/dd/yyyy
        df_precio['Date'] = df_precio['Date'].dt.strftime('%m/%d/%Y')
        df_indicador['Date'] = df_indicador['Date'].dt.strftime('%m/%d/%Y')

        precio_dfs[ticker] = df_precio
        indicador_dfs[ticker] = df_indicador

# Exportar a Excel
with pd.ExcelWriter(os.path.join(output_folder, "PrecioHistorico.xlsx")) as writer:
    for ticker, df in precio_dfs.items():
        df.to_excel(writer, sheet_name=ticker, index=False)

with pd.ExcelWriter(os.path.join(output_folder, "IndicadorTecnica.xlsx")) as writer:
    for ticker, df in indicador_dfs.items():
        df.to_excel(writer, sheet_name=ticker, index=False)

print("✅ Excels generados correctamente en la carpeta:", output_folder)
