import pandas as pd

def clean_data(df):
    """
    Limpia un DataFrame eliminando las filas con valores faltantes.

    Parámetros:
        df (pd.DataFrame): El DataFrame que se desea limpiar.

    Retorna:
        pd.DataFrame: Un nuevo DataFrame sin filas que contengan valores nulos.

    Uso:
        Este método es útil para preparar datos eliminando las observaciones incompletas.
    """
    df = df.dropna()
    return df


def normalize_data(df, columns):
    """
    Normaliza los datos en las columnas especificadas del DataFrame.

    La normalización se realiza utilizando la fórmula:
        z = (x - mean) / std_dev

    Parámetros:
        df (pd.DataFrame): El DataFrame con los datos a normalizar.
        columns (list): Lista de nombres de columnas a normalizar.

    Retorna:
        pd.DataFrame: El DataFrame con las columnas normalizadas.

    Uso:
        Este método es útil para ajustar los datos a un rango estándar
        antes de aplicarlos a modelos estadísticos o de machine learning.
    """
    for col in columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df


def remove_multindex(df):
    """
    Elimina múltiples niveles en los índices de columnas de un DataFrame,
    conservando únicamente el primer nivel.

    Parámetros:
        df (pd.DataFrame): El DataFrame con índices de columnas potencialmente multinivel.

    Retorna:
        pd.DataFrame: Un DataFrame con un solo nivel en las columnas.

    Uso:
        Este método es útil cuando se trabaja con datos que tienen índices de columnas complejos,
        como los resultantes de ciertas operaciones en pandas (e.g., pivot tables o groupby).
    """
    remove = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        remove.columns = [col[0] for col in remove.columns]
    return remove

def prepare_data_for_backtrader(df):
    """
    Convierte un DataFrame de yfinance para ser utilizado en Backtrader.
    
    Parámetros:
        df (pd.DataFrame): DataFrame obtenido con la API de yfinance.
                           Debe contener columnas: 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'.
    
    Retorna:
        pd.DataFrame: DataFrame transformado con el índice como datetime y columnas renombradas.
    """
    # Asegurarse de que el índice sea de tipo datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Verificar que las columnas requeridas estén presentes
    required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"El DataFrame debe contener las columnas: {required_columns}")
    
    # Opcionalmente usar 'Adj Close' como 'Close'
    if 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    
    # Seleccionar las columnas necesarias y asegurarse del orden correcto
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Retornar el DataFrame transformado
    return df