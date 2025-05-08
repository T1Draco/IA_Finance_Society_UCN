from .stock_trading_env import StockTradingEnv
import numpy as np

class SmallInvestorEnv(StockTradingEnv):
    def __init__(self, df):
        # Capital inicial reducido
        initial_balance = 1000

        # Llamamos al constructor del entorno base con el nuevo capital
        super().__init__(df, initial_balance=initial_balance)

        # Ajustes específicos del inversor pequeño
        self.transaction_cost = 0.002  # 0.2% comisión
        self.max_shares_per_trade = 3  # compra en lotes chicos

    def step(self, action):
        row = self.df.iloc[self.current_step]
        price = row["Close"]
        prev_total_asset = self.balance + self.shares_held * price
        prev_balance = self.balance
        reward = 0
        reward_bonus = 0

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

                self.shares_held += shares_bought
                self.balance -= total_spent
            else:
                reward -= 0.05  # ❌ penaliza intento de comprar sin cash

        # === Acción: Vender ===
        elif action == 2:
            if self.shares_held > 0:
                cost = self.shares_held * price * self.transaction_cost
                self.balance += self.shares_held * price - cost

                if price > self.avg_buy_price:
                    reward_bonus += 1  # ✅ vendió con ganancia
                else:
                    reward_bonus -= 0.5  # ❌ vendió con pérdida

                self.shares_held = 0
                self.avg_buy_price = 0
            else:
                reward -= 0.1  # ❌ penaliza vender sin tener acciones

        # === Acción: Hold ===
        elif action == 0:
            reward -= 0.01  # leve penalización por inacción continua

        # === Avanza un paso
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # === Calcular nuevo capital y reward base
        new_total_asset = self.balance + self.shares_held * price
        capital_gain = new_total_asset - prev_total_asset
        reward = 0  # reiniciamos reward base (modular)

        # === 🎯 Reward proporcional continuo
        reward += np.tanh(capital_gain / 20)  # escala ajustable

        # === 💰 Bonus por ganancia significativa
        if capital_gain > 10:
            reward += 0.2

        # === 📉 Penalización por pérdida sostenida
        if new_total_asset < prev_total_asset:
            reward -= 0.05  # penaliza pérdida neta en el paso

        # === 🧠 Penalización por mantener sin convicción
        if action in [1, 2] and abs(price - self.avg_buy_price) < 0.5:
            reward -= 0.05

        # === 🧯 Penalización si se queda sin liquidez
        if self.balance < 1:
            reward -= 0.2

        # === 📊 Penalización por drawdown respecto a máximo anterior
        if new_total_asset < self.total_asset:
            reward -= 0.1 + 0.001 * (self.total_asset - new_total_asset)

        # === 🎉 Bonus pequeño si rompe el capital anterior
        if new_total_asset > self.total_asset:
            reward += 0.05

        # === Aplicar bonus/penalizaciones de la acción
        reward += reward_bonus

        # Actualizar asset total
        self.total_asset = new_total_asset
        return self._get_obs(), reward, done, {}


