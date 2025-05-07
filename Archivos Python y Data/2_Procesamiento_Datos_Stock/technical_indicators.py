import pandas as pd

def calcular_sma(df, columna='Close', ventana=20):   # Media Móvil Simple
    return df[columna].rolling(window=ventana).mean()

def calcular_rsi(df, columna='Close', ventana=14):    # Relative Strength Index (Índice de Fuerza Relativa)
    delta = df[columna].diff()
    ganancia = delta.clip(lower=0)
    perdida = -delta.clip(upper=0)
    media_ganancia = ganancia.rolling(window=ventana).mean()
    media_perdida = perdida.rolling(window=ventana).mean()
    rs = media_ganancia / media_perdida
    return 100 - (100 / (1 + rs))

def calcular_macd(df, columna='Close', rapida=12, lenta=26, signal=9):      # Moving Average Convergence Divergence (diferencia entre dos medias móviles exponenciales, una rápida y la otra lenta)
    ema_rapida = df[columna].ewm(span=rapida, adjust=False).mean()
    ema_lenta = df[columna].ewm(span=lenta, adjust=False).mean()
    macd = ema_rapida - ema_lenta
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                      COMPARATIVA DE INDICADORES TÉCNICOS – SMA, RSI, MACD                                    ║
╠════════════╦════════════════════╦══════════════════════════════════╦═════════════════════════════════════════╣
║ Indicador  ║ Tipo               ║ Sensibilidad                     ║ Señales Típicas                         ║
╠════════════╬════════════════════╬══════════════════════════════════╬═════════════════════════════════════════╣
║ SMA        ║ Tendencia          ║ Baja (más estable si ventana ↑)  ║ Cruce de SMA rápida con lenta           ║
║            ║                    ║                                  ║ — SMA corta > SMA larga → compra        ║
║            ║                    ║                                  ║ — SMA corta < SMA larga → venta         ║
╠════════════╬════════════════════╬══════════════════════════════════╬═════════════════════════════════════════╣
║ RSI        ║ Momentum           ║ Media (fijo en 14 días por conv.)║ Sobrecompra / Sobreventa                ║
║            ║                    ║                                  ║ — RSI > 70 → sobrecomprado → venta      ║
║            ║                    ║                                  ║ — RSI < 30 → sobrevendido → compra      ║
║            ║                    ║                                  ║ — RSI cruzando 50 puede marcar tendencia║
╠════════════╬════════════════════╬══════════════════════════════════╬═════════════════════════════════════════╣
║ MACD       ║ Momentum+Tendencia║ Alta (muy sensible a cambios)    ║ Cruces de MACD con línea de señal       ║
║            ║                    ║                                  ║ — MACD > Señal → impulso alcista        ║
║            ║                    ║                                  ║ — MACD < Señal → impulso bajista        ║
║            ║                    ║                                  ║ — MACD cruza eje 0 → cambio de tendencia║
╠════════════╬════════════════════╬══════════════════════════════════╬═════════════════════════════════════════╣
║ Aplicación ║ Todos los activos  ║ Ajustable por volatilidad        ║ Combinables entre sí para mayor robustez║
╚════════════╩════════════════════╩══════════════════════════════════╩═════════════════════════════════════════╝
"""
