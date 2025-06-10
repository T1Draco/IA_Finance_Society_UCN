import os
from glob import glob

# Lista de tickers basados en archivos de modelo
MODELS_DIR = "models/LSTM/models_pytorch"
tickers = [os.path.basename(f).replace("_lstm_multivariable_model.pth", "") for f in glob(f"{MODELS_DIR}/*_lstm_multivariable_model.pth")]

# Ejecutar el script de inferencia para cada ticker
for ticker in tickers:
    print(f"\n🔄 Ejecutando inferencia para {ticker}...")
    os.system(f"python make_preds.py --ticker \"{ticker}\"")
