from .stock_trading_env import StockTradingEnv
import numpy as np

class SmallInvestorEnv(StockTradingEnv):
    def __init__(self, df):
        # Capital inicial reducido
        initial_balance = 1000

        # Llamamos al constructor del entorno base con el nuevo capital
        super().__init__(df, initial_balance=initial_balance)
        self.log = []  # para debug y análisis post-entrenamiento

        # Ajustes específicos del inversor pequeño
        self.transaction_cost = 0.002  # 0.2% comisión
        self.max_shares_per_trade = 3  # compra en lotes chicos
        self.steps_without_action = 0

    def step(self, action):
        row = self.df.iloc[self.current_step - 1]
        row_now = self.df.iloc[self.current_step]
        row_pred = row_now
        price = row_now["Close"]
        prev_total_asset = self.balance + self.shares_held * price

        reward = 0
        reward_bonus = 0
        self.last_shares_bought = 0
        self.last_shares_sold = 0

        # === Acción: Comprar ===
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

                if abs(price - row["SMA_20"]) < 2:
                    reward_bonus += 0.1  # operó cerca de nivel técnico

                self.shares_held += shares_bought
                self.balance -= total_spent
                self.last_shares_bought = shares_bought

                self.steps_without_action = 0

            else:
                reward -= 0.2  # Intento de compra sin saldo

        # === Acción: Vender ===
        elif action == 2:
            if self.shares_held > 0:
                self.last_shares_sold = self.shares_held
                cost = self.shares_held * price * self.transaction_cost
                self.balance += self.shares_held * price - cost

                if price > self.avg_buy_price:
                    profit = (price - self.avg_buy_price) * self.last_shares_sold
                    reward_bonus += 0.4 + np.tanh(profit / 8)
                else:
                    reward_bonus -= 0.4  # penalización por pérdida

                if abs(price - row["SMA_20"]) < 2:
                    reward_bonus += 0.1  # operó cerca de nivel técnico

                if price > self.avg_buy_price:
                    profit = (price - self.avg_buy_price) * self.last_shares_sold
                    reward_bonus += 0.4 + np.tanh(profit / 15)

                    # 🔼 vender cuando hay señal de bajada
                    if row_now["Close"] < row_now["SMA_20"] and row_pred["Pred"] < row_now["Close"]:
                        reward_bonus += 0.2  # venta estratégica

                self.steps_without_action = 0

                if price > self.avg_buy_price and self.shares_held > 0:
                    reward_bonus += 0.6  # incentivo fuerte por vender en ganancia

                if self.shares_held > 0 and row_pred["Pred"] < price:
                    reward_bonus += 0.3  # si se anticipa caída, incentivo por salir

                self.shares_held = 0
                self.avg_buy_price = 0
            else:
                reward -= 0.3  # Vendió sin tener acciones

        # === Acción: Hold ===
        elif action == 0:
            if self.shares_held > 0 and row_now["Close"] > row["Close"] and row_pred["Pred"] > row["Close"]:
                reward_bonus += 0.25  # mayor incentivo por buen hold
            elif self.shares_held > 0 and row_pred["Pred"] < row["Close"] and price < row["Close"]:
                reward_bonus -= 0.2  # mantuvo en caída

        # === Paso del tiempo ===
        self.current_step += 1

        new_total_asset = self.balance + self.shares_held * price
        capital_gain = new_total_asset - prev_total_asset
        reward += np.tanh(capital_gain / 20)

        # Bonus/penalización si supera o cae respecto al capital histórico
        if new_total_asset > self.total_asset:
            reward += 0.5 + np.tanh((new_total_asset - self.total_asset) / 10)
        else:
            reward -= 1 + 0.003 * (self.total_asset - new_total_asset)

        # Penalización por mantener acciones con pérdida
        if self.shares_held > 0 and price < self.avg_buy_price:
            reward -= 0.005 * self.shares_held

        # Penalización por operar sin convicción
        if action in [1, 2] and abs(price - self.avg_buy_price) < 0.5:
            reward -= 0.05

        # Penalización si sin liquidez
        if self.balance < 1:
            reward -= 0.1

        self.steps_without_action += 1
        if self.steps_without_action > 20:
            reward -= 0.05  # penalización leve por inacción prolongada

        if self.total_asset < prev_total_asset and reward > 0:
            reward -= 0.1  # penaliza subir reward cuando el capital baja

        if action in [1, 2] and reward > 0.5:
            reward += 0.1  # pequeña bonificación por operar bien

        # Al final de step()
        if self.shares_held > 10 and price < self.avg_buy_price:
            reward -= 0.05 * (self.shares_held / 10)  # castiga acumular sin control

        # Aplicar acumulado
        reward += reward_bonus
        self.total_asset = new_total_asset

        # === LOG para debug ===
        if not hasattr(self, "log"):
            self.log = []
        self.log.append({
            "step": self.current_step,
            "action": action,
            "reward": reward,
            "reward_bonus": reward_bonus,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "price": price,
            "avg_buy_price": self.avg_buy_price,
            "capital_gain": capital_gain,
            "total_asset": self.total_asset
        })

        terminated = self.current_step >= len(self.df) - 1
        truncated = False  # puedes usar alguna lógica aquí si quieres cortar por otras razones
        return self._get_obs(), reward, terminated, truncated, {}


