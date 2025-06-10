import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    def __init__(self, df,
                 initial_balance=1000,
                 transaction_cost=0.001,
                 max_shares_per_trade=10,
                 reward_fn=None):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_shares_per_trade = max_shares_per_trade
        self.reward_fn = reward_fn  # Nueva: permite usar lógica externa

        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
            dtype=np.float32
        )

        self.seed()
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
        self.last_shares_bought = 0
        self.last_shares_sold = 0
        self.steps_without_action = 0

        return self._get_obs(), {}

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
        prev_total_asset = self.balance + self.shares_held * price

        # === Inicializar costo y reward
        cost = 0
        reward = 0
        self.last_shares_bought = 0
        self.last_shares_sold = 0

        # === Acción: Comprar
        if action == 1:
            shares_bought = min(self.balance // price, self.max_shares_per_trade)
            if shares_bought > 0:
                cost = shares_bought * price * self.transaction_cost
                total_spent = shares_bought * price + cost

                total_shares = self.shares_held + shares_bought
                if total_shares > 0:
                    self.avg_buy_price = (
                            (self.avg_buy_price * self.shares_held + price * shares_bought) / total_shares
                    )

                self.shares_held += shares_bought
                self.balance -= total_spent
                self.last_shares_bought = shares_bought

            else:
                reward -= 0.1  # Penalización por intento de compra sin fondos

        # === Acción: Vender
        elif action == 2:
            if self.shares_held > 0:
                self.last_shares_sold = self.shares_held
                cost = self.shares_held * price * self.transaction_cost
                self.balance += self.shares_held * price - cost
                self.shares_held = 0
                self.avg_buy_price = 0
            else:
                reward -= 0.2  # Penalización por intentar vender sin tener acciones

        # === Acción: Hold
        elif action == 0:
            reward -= 0.005  # Leve penalización por inacción

        # === Avanza paso
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        new_total_asset = self.balance + self.shares_held * price
        capital_gain = new_total_asset - prev_total_asset
        reward += np.tanh(capital_gain / 20)  # reward proporcional suave
        self.total_asset = new_total_asset

        # === Recompensa personalizada (si existe)
        if self.reward_fn is not None:
            row_prev = self.df.iloc[self.current_step - 2] if self.current_step > 1 else row
            reward += self.reward_fn(self, action, row, price, capital_gain)
        obs = self._get_obs()
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Shares held: {self.shares_held}, Asset: {self.total_asset}")
