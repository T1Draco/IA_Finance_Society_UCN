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
        row = self.df.iloc[self.current_step - 1]  # ayer
        row_now = self.df.iloc[self.current_step]  # hoy
        row_pred = row_now  # predicción del día (ya disponible)
        price = row_now["Close"]
        prev_total_asset = self.balance + self.shares_held * price
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

                # ✅ Compró bajo SMA
                if price < row["SMA_20"]:
                    reward_bonus += 0.2

                # ❌ Compró con poca liquidez restante
                if self.balance < price * self.max_shares_per_trade * 1.1:
                    reward -= 0.3

                # ❌ Compró en condiciones bajistas
                if row_pred["Pred"] <= row["Close"] and price < row["Close"]:
                    reward -= 0.4

            else:
                reward -= 0.05  # ❌ intentó comprar sin cash

        # === Acción: Vender ===
        elif action == 2:
            if self.shares_held > 0:
                cost = self.shares_held * price * self.transaction_cost
                self.balance += self.shares_held * price - cost

                # ✅ Vendió con ganancia
                if price > self.avg_buy_price:
                    reward_bonus += 1.0
                    if (price - self.avg_buy_price) > 3:
                        reward_bonus += 0.3  # gran ganancia

                else:
                    reward_bonus -= 0.5  # ❌ vendió con pérdida

                if price > row["SMA_20"]:
                    reward_bonus += 0.1  # vendió alto

                self.shares_held = 0
                self.avg_buy_price = 0

                # ✅ Salió antes de una caída
                if row_pred["Pred"] < row["Close"] and price < row["Close"]:
                    reward_bonus += 0.4

                # ❌ Vendió antes de una subida
                if row_pred["Pred"] > row["Close"] and price > row["Close"]:
                    reward_bonus -= 0.3

            else:
                reward -= 0.1  # ❌ vendió sin tener

        # === Acción: Hold ===
        elif action == 0:
            price_change = row_now["Close"] - row["Close"]
            if price_change > 2 and self.shares_held == 0:
                reward -= 0.1  # ❌ no compró en subida
            elif price_change < -2 and self.shares_held > 0:
                reward -= 0.1  # ❌ no vendió en bajada
            else:
                reward -= 0.005

            # ❌ mantuvo muchas acciones durante caída con mala predicción
            if self.shares_held > 0 and row_pred["Pred"] < row["Close"] and price < row["Close"]:
                reward -= 0.3

            # ❌ volatilidad alta sin acción
            volatility = abs(row["Close"] - row["SMA_20"])
            if volatility > 3:
                reward -= 0.1

        # === Avanza un paso
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # === Calcular reward base
        new_total_asset = self.balance + self.shares_held * price
        capital_gain = new_total_asset - prev_total_asset
        reward += np.tanh(capital_gain / 20)

        # === Penalización por pérdidas
        if new_total_asset < prev_total_asset:
            reward -= 0.05
        if new_total_asset < self.total_asset:
            reward -= 0.1 + 0.001 * (self.total_asset - new_total_asset)

        # === Bonus por superar capital previo
        if new_total_asset > self.total_asset:
            reward += 0.05

        # === Penalización si sin liquidez
        if self.balance < 1:
            reward -= 0.2

        # === Penalización por operar sin convicción
        if action in [1, 2] and abs(price - self.avg_buy_price) < 0.5:
            reward -= 0.05

        # ❌ Mantener acciones con pérdida penaliza
        if self.shares_held > 0 and price < self.avg_buy_price:
            reward -= 0.01 * self.shares_held

        # Aplicar bonus/penalización acumulada
        reward += reward_bonus

        # Actualizar total asset
        self.total_asset = new_total_asset

        return self._get_obs(), reward, done, {}
