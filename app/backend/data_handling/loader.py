import pandas as pd

def load_file_from_streamlit(uploaded_file):
    """
    Carga un archivo CSV o Excel desde un objeto subido con Streamlit.

    Parámetros:
        uploaded_file (io.BytesIO): Archivo subido mediante Streamlit.

    Retorna:
        pd.DataFrame: Datos cargados en un DataFrame.

    Lanza:
        ValueError: Si el formato del archivo no es soportado.
    """
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx") or uploaded_file.name.endswith(".xls"):
        import openpyxl
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Formato de archivo no soportado. Use un archivo CSV o Excel.")

def load_yfinance_data(ticker, start, end):
    """
    Carga los datos historicos para un ticker y fechas seleccionadas

    Parámetros:
        ticker: Nombre de la serie.
        start: Fecha de inicio.
        end: Fecha de termino.

    Retorna:
        pd.DataFrame: Datos cargados en un DataFrame.

    Lanza:
        ValueError: Si las fechas son incorrectas.
    """
    import yfinance as yf
    # Validar fechas
    if start >= end:
        raise ValueError("La fecha de inicio debe ser anterior a la fecha de fin.")
    data = yf.download(ticker, start=start, end=end)
    return data

def load_backtrader_strategy(uploaded_file):
    """
    Carga y valida un archivo Python como estrategia para Backtrader.

    Parámetros:
        uploaded_file (io.BytesIO): Archivo subido desde `st.file_uploader`.

    Retorna:
        class: La clase de estrategia cargada.

    Lanza:
        ValueError: Si el archivo no tiene una extensión .py o no contiene una estrategia válida.
        ImportError: Si ocurre un error al cargar el módulo.
    """
    import importlib.util
    import sys
    import tempfile
    import backtrader as bt

    # Verificar que el archivo tiene extensión .py
    if not uploaded_file.name.endswith(".py"):
        raise ValueError("El archivo debe ser un script de Python con extensión .py")

    # Crear un archivo temporal en el sistema
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Cargar el módulo dinámicamente
    module_name = uploaded_file.name[:-3]  # Elimina la extensión .py para el nombre del módulo
    spec = importlib.util.spec_from_file_location(module_name, temp_file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        # Agregar el módulo al sistema de módulos importados
        sys.modules[module_name] = module
    except Exception as e:
        raise ImportError(f"Error al cargar el módulo: {e}")

    # Buscar clases que hereden de `bt.Strategy`
    strategy_classes = [
        obj for name, obj in vars(module).items()
        if isinstance(obj, type) and issubclass(obj, bt.Strategy) and obj is not bt.Strategy
    ]

    if not strategy_classes:
        raise ValueError("El módulo no contiene una estrategia válida que herede de `bt.Strategy`.")

    # Retornar la primera estrategia encontrada
    return strategy_classes[0]

def load_multiple_strategies(uploaded_files):
    """
    Carga y valida múltiples archivos Python como estrategias para Backtrader.

    Parámetros:
        uploaded_files (list): Lista de archivos subidos desde `st.file_uploader`.

    Retorna:
        dict: Un diccionario donde las claves son los nombres de los archivos y los valores son listas
              de clases de estrategias válidas encontradas en cada archivo.

    Maneja:
        ValueError: Si algún archivo no tiene una extensión .py o no contiene estrategias válidas.
        ImportError: Si ocurre un error al cargar alguno de los módulos.
    """
    strategies = {}

    for uploaded_file in uploaded_files:
        try:
            # Cargar la estrategia usando la función individual
            strategy_class = load_backtrader_strategy(uploaded_file)
            
            # Guardar la clase completa en el diccionario
            strategies[uploaded_file.name] = strategy_class
        
        except ValueError as ve:
            print(f"Error en el archivo {uploaded_file.name}: {ve}")
        
        except ImportError as ie:
            print(f"Error al importar el módulo desde {uploaded_file.name}: {ie}")

    return strategies
