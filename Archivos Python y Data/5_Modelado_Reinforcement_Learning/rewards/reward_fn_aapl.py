import numpy as np

def reward_fn_aapl(env, action, row, price, capital_gain):
    reward = 0
    reward_bonus = 0

    row_now = env.df.iloc[env.current_step]
    row_pred = row_now  # se puede cambiar si tienes predicción separada

    # === Acción: Comprar ===
    if action == 1:
        if env.last_shares_bought > 0:
            if price < row["SMA_20"]:
                reward_bonus += 0.2
            if env.balance < price * env.last_shares_bought * 1.1:
                reward -= 0.3
            if row_pred["Pred"] <= row["Close"] and price < row["Close"]:
                reward -= 0.4
        else:
            reward -= 0.2

    # === Acción: Vender ===
    elif action == 2:
        if env.last_shares_sold > 0:
            if price > env.avg_buy_price:
                reward_bonus += 1.0
                if (price - env.avg_buy_price) > 3:
                    reward_bonus += 0.3
            else:
                reward_bonus -= 0.5

            if price > row["SMA_20"]:
                reward_bonus += 0.1

            if row_pred["Pred"] < row["Close"] and price < row["Close"]:
                reward_bonus += 0.4
            if row_pred["Pred"] > row["Close"] and price > row["Close"]:
                reward_bonus -= 0.3
        else:
            reward -= 0.2

    # === Acción: Hold ===
    elif action == 0:
        price_change = row_now["Close"] - row["Close"]
        if price_change > 2 and env.shares_held == 0:
            reward -= 0.1
        elif price_change < -2 and env.shares_held > 0:
            reward -= 0.1
        else:
            reward -= 0.005

        if env.shares_held > 0 and row_pred["Pred"] < row["Close"] and price < row["Close"]:
            reward -= 0.3

        volatility = abs(row["Close"] - row["SMA_20"])
        if volatility > 3:
            reward -= 0.1

    # === Recompensa base por ganancia de capital
    reward += np.tanh(capital_gain / 20)

    # Penalización por pérdidas
    if env.total_asset < env.total_asset - capital_gain:
        reward -= 0.05
    if env.total_asset < env.total_asset - 1:
        reward -= 0.1 + 0.001 * (env.total_asset - (env.total_asset - capital_gain))

    # Bonus por superar capital previo
    if capital_gain > 0:
        reward += 0.05

    # Penalización si sin liquidez
    if env.balance < 1:
        reward -= 0.2

    # Penalización por operar sin convicción
    if action in [1, 2] and abs(price - env.avg_buy_price) < 0.5:
        reward -= 0.05

    # Penalización por mantener acciones en pérdida
    if env.shares_held > 0 and price < env.avg_buy_price:
        reward -= 0.01 * env.shares_held

    reward += reward_bonus
    return reward
