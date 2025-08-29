import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class FractionalStockTradingEnv(gym.Env):
    def __init__(self, df,
                 initial_balance=1000,
                 transaction_cost=0.001,
                 max_position_pct=0.95,  # Máximo % del balance para una operación
                 min_trade_amount=10,  # Mínimo $ para una operación
                 reward_fn=None):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_pct = max_position_pct
        self.min_trade_amount = min_trade_amount
        self.reward_fn = reward_fn

        # Acción continua: [-1, 1] donde:
        # -1 = vender todo, 0 = hold, 1 = comprar máximo permitido
        # Valores intermedios = operaciones parciales
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),  # Agregamos posición actual
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
        self.shares_held = 0.0  # Ahora puede ser fraccionario
        self.current_step = 0
        self.total_asset = self.balance
        self.avg_buy_price = 0.0
        self.last_shares_bought = 0.0
        self.last_shares_sold = 0.0
        self.last_trade_value = 0.0
        self.position_pct = 0.0  # % del portafolio en acciones

        return self._get_obs(), {}

    def _get_obs(self):
        if self.current_step == 0:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            obs[0] = self.df.iloc[0]["Pred"]
            return obs

        row = self.df.iloc[self.current_step - 1]
        row_pred = self.df.iloc[self.current_step]

        # Calcular % de posición actual
        current_price = row["Close"]
        position_value = self.shares_held * current_price
        total_portfolio = self.balance + position_value
        self.position_pct = position_value / total_portfolio if total_portfolio > 0 else 0

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
            row["sma_diff"],
            self.position_pct  # Nueva feature: % de posición
        ], dtype=np.float32)

        return obs

    def step(self, action):
        row = self.df.iloc[self.current_step]
        price = row["Close"]
        prev_total_asset = self.balance + self.shares_held * price

        # Resetear variables de la operación anterior
        cost = 0
        reward = 0
        self.last_shares_bought = 0.0
        self.last_shares_sold = 0.0
        self.last_trade_value = 0.0

        # Convertir acción continua a operación específica
        action_value = float(action[0])

        # === Lógica de operación fraccionaria ===
        if action_value > 0.1:  # COMPRAR (threshold para evitar micro-operaciones)
            # Calcular cuánto dinero usar para comprar
            available_balance = self.balance
            max_trade_value = available_balance * self.max_position_pct
            trade_value = max_trade_value * action_value  # Escalar por intensidad de la acción

            if trade_value >= self.min_trade_amount and trade_value <= available_balance:
                shares_to_buy = trade_value / price
                cost = trade_value * self.transaction_cost
                total_spent = trade_value + cost

                if total_spent <= self.balance:
                    # Actualizar precio promedio de compra
                    total_shares_after = self.shares_held + shares_to_buy
                    if total_shares_after > 0:
                        self.avg_buy_price = (
                                (self.avg_buy_price * self.shares_held + price * shares_to_buy) / total_shares_after
                        )

                    self.shares_held += shares_to_buy
                    self.balance -= total_spent
                    self.last_shares_bought = shares_to_buy
                    self.last_trade_value = trade_value
            else:
                reward -= 0.05  # Penalización leve por intento de compra inválido

        elif action_value < -0.1:  # VENDER (threshold para evitar micro-operaciones)
            if self.shares_held > 0:
                # Calcular cuántas acciones vender
                sell_pct = abs(action_value)  # -1 = vender todo, -0.5 = vender 50%
                shares_to_sell = self.shares_held * sell_pct
                trade_value = shares_to_sell * price

                if trade_value >= self.min_trade_amount:
                    cost = trade_value * self.transaction_cost
                    self.balance += trade_value - cost
                    self.shares_held -= shares_to_sell
                    self.last_shares_sold = shares_to_sell
                    self.last_trade_value = trade_value

                    # Si vendió todo, resetear precio promedio
                    if self.shares_held < 0.001:  # Prácticamente cero
                        self.shares_held = 0.0
                        self.avg_buy_price = 0.0
            else:
                reward -= 0.05  # Penalización por intentar vender sin tener

        # === Avanzar paso ===
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        new_total_asset = self.balance + self.shares_held * price
        capital_gain = new_total_asset - prev_total_asset
        reward += np.tanh(capital_gain / 20)  # Reward base proporcional
        self.total_asset = new_total_asset

        # === Recompensa personalizada (si existe) ===
        if self.reward_fn is not None:
            reward += self.reward_fn(self, action_value, row, price, capital_gain)

        obs = self._get_obs()
        info = {
            'balance': self.balance,
            'shares_held': self.shares_held,
            'total_asset': self.total_asset,
            'last_trade_value': self.last_trade_value,
            'position_pct': self.position_pct
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        position_value = self.shares_held * self.df.iloc[min(self.current_step, len(self.df) - 1)]["Close"]
        print(f"Step: {self.current_step}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Shares held: {self.shares_held:.4f}")
        print(f"Position value: ${position_value:.2f}")
        print(f"Total asset: ${self.total_asset:.2f}")
        print(f"Position %: {self.position_pct * 100:.1f}%")
        print("-" * 40)