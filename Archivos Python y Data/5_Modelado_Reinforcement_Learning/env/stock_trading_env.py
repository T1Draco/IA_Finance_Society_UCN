import gym
from gym import spaces
import numpy as np
import pandas as pd


class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(StockTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = 0.001  # 0.1% (más estándar)

        # Acción: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

        # Observación: predicción LSTM y precio real
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.total_asset = self.balance
        self.avg_buy_price = 0  # 🆕 nuevo atributo para seguimiento de compras
        return self._get_obs()

    def _get_obs(self):
        row = self.df.iloc[self.current_step]

        obs = np.array([
            row["Close"],
            self.balance,
            self.shares_held,
            self.avg_buy_price,
            (row["Close"] - self.avg_buy_price) if self.shares_held > 0 else 0,
            row["retorno_simple"],
            row["retorno_log"],
            row["SMA_20"],
            row["SMA_50"],
            row["RSI_14"],
            row["MACD"],
            row["MACD_signal"],
            row["sma_diff"]
        ], dtype=np.float32)

        return obs

    def step(self, action):
        row = self.df.iloc[self.current_step]
        price = row['Close']
        cost = 0

        if action == 1:  # Buy
            shares_bought = self.balance // price
            cost = shares_bought * price * self.transaction_cost
            self.shares_held += shares_bought
            self.balance -= (shares_bought * price + cost)

        elif action == 2 and self.shares_held > 0:  # Sell
            cost = self.shares_held * price * self.transaction_cost
            self.balance += self.shares_held * price - cost
            self.shares_held = 0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        next_total_asset = self.balance + self.shares_held * price
        reward = next_total_asset - self.total_asset - cost
        self.total_asset = next_total_asset

        return self._get_obs(), reward, done, {}

    def render(self):
        print(
            f"Step: {self.current_step}, Balance: {self.balance}, Shares held: {self.shares_held}, Asset: {self.total_asset}")
