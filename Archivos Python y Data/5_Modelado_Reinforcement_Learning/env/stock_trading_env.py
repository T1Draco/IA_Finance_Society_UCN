import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Box


class SimpleTradingEnv(Env):
    def __init__(self, data_path, initial_balance=10000):
        super(SimpleTradingEnv, self).__init__()

        # Cargar datos del CSV preparado
        self.df = pd.read_csv(data_path, parse_dates=['Date'])
        self.df = self.df.sort_values('Date').reset_index(drop=True)

        # Features: Indicadores y Pred (Pred es para t, basado en t-1)
        self.features = ['High', 'Low', 'Open', 'Volume', 'retorno_simple', 'retorno_log',
                         'SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_signal', 'Pred']

        # Calcular rangos aproximados basados en los datos (ajústalos según tus datos)
        data_sample = self.df[self.features].values
        low_bound = np.min(data_sample) - 1.0  # Margen pequeño
        high_bound = np.max(data_sample) + 1.0  # Margen pequeño

        # Espacio de observación: features (de t-1) + shares_held norm + balance norm
        obs_size = len(self.features) + 2
        self.observation_space = Box(low=low_bound, high=high_bound, shape=(obs_size,), dtype=np.float32)

        # Espacio de acción: [-1, 1] para proporción buy/sell
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.initial_balance = initial_balance
        self.min_step = 1  # Empezar desde el segundo día para tener datos de t-1
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.min_step
        self.balance = self.initial_balance
        # Inicializar shares_held con 10% del balance inicial basado en el precio del primer paso
        initial_price = self.df.iloc[self.min_step - 1]['Close']
        if initial_price <= 0:
            raise ValueError("El precio inicial es inválido (menor o igual a 0)")
        self.shares_held = (self.initial_balance * 0.1) / initial_price  # 10% del balance en acciones
        self.net_worth = self.balance + self.shares_held * initial_price
        self.previous_net_worth = self.net_worth
        observation = self._get_observation()
        info = {}  # Diccionario vacío para info adicional
        return observation, info  # Devolver explícitamente la tupla

    def _get_observation(self):
        # Siempre usa datos de t-1 (evita leakage)
        data = self.df.iloc[self.current_step - 1][self.features].values.astype(np.float32)

        # Normalizar shares_held y balance (dividido por initial_balance)
        current_price = self.df.iloc[self.current_step - 1]['Close']  # Precio de t-1 para normalización
        norm_shares = (self.shares_held * current_price) / self.initial_balance
        norm_balance = self.balance / self.initial_balance

        obs = np.concatenate([data, [norm_shares, norm_balance]])
        return obs

    def step(self, action):
        action = action[0]  # Escalar acción

        # Precio actual (Close_t) para ejecutar trade
        current_price = self.df.iloc[self.current_step]['Close']

        # Ejecutar acción (sin costos de transacción por simplicidad)
        if action > 0:  # Buy: proporción del balance
            max_buy = self.balance / current_price
            shares_to_buy = max_buy * action
            cost = shares_to_buy * current_price
            self.shares_held += shares_to_buy
            self.balance -= cost
        elif action < 0:  # Sell: proporción de shares_held
            shares_to_sell = self.shares_held * abs(action)
            revenue = shares_to_sell * current_price
            self.shares_held -= shares_to_sell
            self.balance += revenue

        # Actualizar net_worth
        self.net_worth = self.balance + self.shares_held * current_price

        # Reward simple: cambio incremental en net_worth
        reward = (self.net_worth - self.previous_net_worth) / self.initial_balance  # Normalización por balance inicial
        self.previous_net_worth = self.net_worth

        # Avanzar paso
        self.current_step += 1
        terminated = self.current_step >= len(self.df)
        truncated = False  # No se usa truncation en este caso simple

        info = {'net_worth': self.net_worth, 'step': self.current_step}

        return self._get_observation(), reward, terminated, truncated, info