@echo off
echo === Ejecutando descarga de datos históricos ===

REM Activar entorno virtual (ajusta esta ruta si tu venv está en otro lado)
call C:\Users\Admin\PycharmProjects\IA_Finance_Society_UCN\3.12_IA_Finance_Society_UCN.venv\Scripts\activate.bat

REM Ejecutar el script principal
python "C:\Users\Admin\PycharmProjects\IA_Finance_Society_UCN\Archivos Python y Data\1. Recolección de Datos\stock_data\download_historical_stocks_price.py"

echo === Proceso terminado ===
pause
