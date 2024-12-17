import backtrader as bt

class MovingAverageCrossStrategy(bt.Strategy):
    """
    Estrategia basada en el cruce de medias móviles.
    Compra cuando la media móvil rápida cruza por encima de la media lenta.
    Vende cuando la media rápida cruza por debajo de la media lenta.
    """

    params = (
        ('fast_period', 30),  # Período de la media móvil rápida
        ('slow_period', 50),  # Período de la media móvil lenta
    )

    def __init__(self):
        # Inicializar las medias móviles
        self.fast_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.slow_period)
        
        # Indicador para detectar cruces
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        # Si la media rápida cruza por encima de la lenta, genera una compra
        if self.crossover > 0:
            if not self.position:  # Solo compra si no hay una posición abierta
                self.buy()
                print(f"Compra en {self.data.close[0]}")

        # Si la media rápida cruza por debajo de la lenta, genera una venta
        elif self.crossover < 0:
            if self.position:  # Solo vende si hay una posición abierta
                self.sell()
                print(f"Venta en {self.data.close[0]}")
