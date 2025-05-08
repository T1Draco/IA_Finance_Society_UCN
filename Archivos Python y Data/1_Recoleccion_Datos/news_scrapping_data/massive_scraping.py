from datetime import datetime, timedelta
from execute_scrapers import ejecutar_scrapers

TICKERS = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "AMZN": "Amazon"
}

fecha_inicio = datetime.strptime("06/05/2023", "%d/%m/%Y")
fecha_fin = datetime.strptime("06/05/2025", "%d/%m/%Y")

fecha_actual = fecha_inicio
total_dias = (fecha_fin - fecha_inicio).days + 1

for i in range(total_dias):
    fecha_str = fecha_actual.strftime("%d/%m/%Y")
    print(f"\n📅 Día {i+1}/{total_dias}: {fecha_str}")

    for ticker, empresa in TICKERS.items():
        try:
            ejecutar_scrapers(ticker, empresa, fecha_str, fecha_str, save_dir="diarios_json")
        except Exception as e:
            print(f"[⚠️] Error en {ticker} el {fecha_str}: {e}")

    fecha_actual += timedelta(days=1)
