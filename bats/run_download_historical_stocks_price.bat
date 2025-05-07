@echo off
echo === Ejecutando descarga de datos historicos ===

REM Activar entorno virtual
call C:\Users\Admin\PycharmProjects\IA_Finance_Society_UCN\3.12_IA_Finance_Society_UCN.venv

REM Cambiar al directorio donde está el script
cd /d "C:\Users\Admin\PycharmProjects\IA_Finance_Society_UCN\Archivos Python y Data\1_Recoleccion_Datos\stock_data"

REM Ejecutar el script
python download_historical_stocks_price.py

echo === Proceso terminado ===
pause
