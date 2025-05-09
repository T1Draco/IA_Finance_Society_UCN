import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Config
ticker = "AAPL"

# Rutas
FORECAST_PATH = f"../../results/predictions/lstm_forecast/{ticker}_forecast.csv"
REAL_DATA_PATH = f"../../../1_Recoleccion_Datos/stock_data/raw_data/{ticker}.csv"

# Cargar datos
df_forecast = pd.read_csv(FORECAST_PATH, parse_dates=["Date"])
df_real = pd.read_csv(REAL_DATA_PATH, parse_dates=["Date"])

# Merge
df_merged = df_forecast.merge(df_real, on="Date", how="inner")

# Métricas
if not df_merged.empty:
    mse = mean_squared_error(df_merged["Close"], df_merged["Forecast"])
    mae = mean_absolute_error(df_merged["Close"], df_merged["Forecast"])
    r2 = r2_score(df_merged["Close"], df_merged["Forecast"])
    print("\n📊 Evaluación de forecast:")
    print(f"   ✅ RMSE : {np.sqrt(mse):.4f}")
    print(f"   ✅ MAE  : {mae:.4f}")
    print(f"   ✅ R²   : {r2:.4f}")
else:
    print("⏳ Aún no hay datos reales para evaluar el forecast.")
