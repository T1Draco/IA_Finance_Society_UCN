import os
import pandas as pd

# Ruta relativa a la carpeta de reportes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Cargar todos los reportes
reportes = []
for archivo in os.listdir(REPORTS_DIR):
    if archivo.endswith("_reporte.csv"):
        df = pd.read_csv(os.path.join(REPORTS_DIR, archivo))
        reportes.append(df)

# Unir todos los reportes en una tabla
df_reportes = pd.concat(reportes, ignore_index=True)

# Mostrar resumen por consola
print("\n COMPARACIÓN DE STOCKS HISTÓRICOS:\n")
print(df_reportes.to_string(index=False))

# Opcional: guardar tabla comparativa como CSV
df_reportes.to_csv(os.path.join(REPORTS_DIR, "resumen_comparativo.csv"), index=False)
print(f"\n Tabla comparativa guardada como: {os.path.join('reports', 'resumen_comparativo.csv')}")


'''
| Métrica                    | Qué representa                                                       |
| -------------------------- | -------------------------------------------------------------------- |
| `Retorno total (%)`        | Cuánto creció (o cayó) el precio desde el primer día hasta el último |
| `Retorno medio diario (%)` | Promedio de cambio porcentual diario (no compuesto)                  |
| `Volatilidad diaria (%)`   | Riesgo histórico (desviación estándar de los retornos diarios)       |
| `Drawdown máximo (%)`      | La peor caída desde un pico histórico                                |
| `Volumen promedio/máximo`  | Liquidez histórica                                                   |
| `RSI promedio/máx/mín`     | Comportamiento típico del indicador RSI en el pasado                 |
| `Precio máximo y mínimo`   | Rango de precios históricos                                          |
'''