import backtrader as bt
import pandas as pd
from backend.data_handling.preprocesing import prepare_data_for_backtrader
from backend.data_handling.preprocesing import remove_multindex


def run_backtest_with_strategies(strategy_dict, data_feed, cash=10000.0, commission=0.001, plot=False):
    """
    Ejecuta un backtest con múltiples estrategias en un dataset dado.

    Parámetros:
        strategy_dict (dict): Un diccionario donde las claves son nombres de archivos y los valores son
                              clases de estrategias que heredan de bt.Strategy.
        data_feed (bt.feeds.DataBase): El feed de datos (por ejemplo, YahooFinanceData o PandasData).
        cash (float): Capital inicial para el broker.
        commission (float): Comisión para cada operación.
        plot (bool): Si True, grafica los resultados al final.

    Retorna:
        dict: Un diccionario con los nombres de los archivos y sus métricas (capital final, drawdown, etc.).
    """

    # Convertir DataFrame a un feed de Backtrader si es necesario
    if isinstance(data_feed, pd.DataFrame):
        df = remove_multindex(data_feed)
        df = prepare_data_for_backtrader(df)

        data_feed = bt.feeds.PandasData(dataname=df)

    # Diccionario para guardar los resultados
    results = {}

    for file_name, strategy in strategy_dict.items():
        # Validar que cada valor en el diccionario sea una clase que herede de bt.Strategy
        if not isinstance(strategy, type) or not issubclass(strategy, bt.Strategy):
            raise ValueError(f"El valor para {file_name} no es una clase válida que herede de bt.Strategy.")

        # Crear cerebro para cada estrategia
        cerebro = bt.Cerebro()
        cerebro.addstrategy(strategy)

        # Agregar los datos
        cerebro.adddata(data_feed)

        # Configurar el broker
        cerebro.broker.set_cash(cash)
        cerebro.broker.setcommission(commission=commission)

        # Configurar analizadores
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")

        # Ejecutar el backtest
        print(f"Ejecutando backtest para la estrategia en {file_name}: {strategy.__name__}")
        strategy_result = cerebro.run()

        # Obtener los analizadores
        drawdown = strategy_result[0].analyzers.drawdown.get_analysis()
        trade_analysis = strategy_result[0].analyzers.trade_analyzer.get_analysis()
        pyfolio_analyzer = strategy_result[0].analyzers.pyfolio

        # Exportar datos de PyFolio para análisis externo
        pyfolio_data = pyfolio_analyzer.get_pf_items()

        # Verificar si pyfolio_data es un diccionario o una tupla
        if isinstance(pyfolio_data, tuple):
            returns_series = pyfolio_data[0]  # Normalmente los retornos están en el índice 0
        elif isinstance(pyfolio_data, dict):
            returns_series = pyfolio_data.get('returns', None)
        else:
            returns_series = None

        if returns_series is None:
            print(f"Advertencia: No se encontraron retornos para la estrategia en {file_name}")

        # Guardar el resultado del capital final y métricas adicionales
        final_value = cerebro.broker.getvalue()
        results[file_name] = {
            "final_value": final_value,
            "drawdown": drawdown,
            "trade_analysis": trade_analysis,
            "pyfolio": pyfolio_data,
            "daily_returns": returns_series,  # Retornos diarios para QuantStats es un pandaSeries
        }

        print(f"Capital final para la estrategia en {file_name}: {final_value:.2f}")

        # Opcional: graficar los resultados
        if plot:
            cerebro.plot()

    return results
