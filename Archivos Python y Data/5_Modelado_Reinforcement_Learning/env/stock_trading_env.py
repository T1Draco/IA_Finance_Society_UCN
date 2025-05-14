import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = 0.001

        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
            dtype=np.float32
        )

        self.seed()  # Inicializa semilla
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.seed(seed)

        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = 0
        self.total_asset = self.balance
        self.avg_buy_price = 0

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        if self.current_step == 0:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            obs[0] = self.df.iloc[0]["Pred"]
            return obs

        row = self.df.iloc[self.current_step - 1]
        row_pred = self.df.iloc[self.current_step]

        obs = np.array([
            row_pred["Pred"],
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
        price = row["Close"]
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
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        next_total_asset = self.balance + self.shares_held * price
        reward = next_total_asset - self.total_asset - cost
        self.total_asset = next_total_asset

        obs = self._get_obs()
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Shares held: {self.shares_held}, Asset: {self.total_asset}")
