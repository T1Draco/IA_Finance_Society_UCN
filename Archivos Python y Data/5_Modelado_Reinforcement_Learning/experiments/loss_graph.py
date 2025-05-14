import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURACIÓN ===
TICKER = "TSLA"
SMOOTHING_WINDOW = 50

# === Rutas del proyecto ===
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "results", "models")
PLOT_DIR = os.path.join(ROOT_DIR, "results", "plots", TICKER)
os.makedirs(PLOT_DIR, exist_ok=True)

# Asegurar que el proyecto raíz esté en sys.path (por si se usan imports locales)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# === Cargar pérdidas ===
loss_path = os.path.join(MODEL_DIR, f"{TICKER}_losses.npy")
losses = np.load(loss_path)

# === Graficar ===
plt.figure(figsize=(12, 6))
plt.plot(losses, label="Loss")
plt.plot(pd.Series(losses).rolling(SMOOTHING_WINDOW).mean(), label=f"Smoothed ({SMOOTHING_WINDOW})", linewidth=2)
plt.title(f"Evolución de la pérdida durante el entrenamiento - {TICKER}")
plt.xlabel("Paso de entrenamiento")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

# === Guardar gráfico ===
plot_path = os.path.join(PLOT_DIR, f"{TICKER}_loss_plot.png")
plt.savefig(plot_path)
plt.close()

# === Confirmación por consola ===
print(f"[📉] Gráfico de pérdidas guardado en: {plot_path}")
