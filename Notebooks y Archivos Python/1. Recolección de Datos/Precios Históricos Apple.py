import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Evita el backend conflictivo
import matplotlib.pyplot as plt

# Descargar datos
df = yf.download("AAPL", start="2022-01-01", end="2025-04-14")

# Ver estructura
print(df.head())

# Graficar cierre
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Close"], label="Precio de cierre", color="blue")
plt.title("Precio de Cierre de Apple en 2022")
plt.xlabel("Fecha")
plt.ylabel("Precio (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
