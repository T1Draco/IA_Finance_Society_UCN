import numpy as np

def reward_fn_tsla(env, action, row, price, capital_gain):
    reward = 0
    reward_bonus = 0

    row_now = env.df.iloc[env.current_step]
    row_pred = row_now  # puedes cambiar esto si luego usas predicción externa

    # === Acción: Comprar ===
    if action == 1:
        if getattr(env, "last_shares_bought", 0) > 0:
            if abs(price - row["SMA_20"]) < 2:
                reward_bonus += 0.1
        else:
            reward -= 0.2

    # === Acción: Vender ===
    elif action == 2:
        if getattr(env, "last_shares_sold", 0) > 0:
            if price > env.avg_buy_price:
                profit = (price - env.avg_buy_price) * env.last_shares_sold
                reward_bonus += 0.4 + np.tanh(profit / 8)
            else:
                reward_bonus -= 0.4

            if abs(price - row["SMA_20"]) < 2:
                reward_bonus += 0.1

            if price > env.avg_buy_price:
                profit = (price - env.avg_buy_price) * env.last_shares_sold
                reward_bonus += 0.4 + np.tanh(profit / 15)

                if row_now["Close"] < row_now["SMA_20"] and row_pred["Pred"] < row_now["Close"]:
                    reward_bonus += 0.2

            if price > env.avg_buy_price and env.shares_held > 0:
                reward_bonus += 0.6

            if env.shares_held > 0 and row_pred["Pred"] < price:
                reward_bonus += 0.3
        else:
            reward -= 0.3

    # === Acción: Hold ===
    elif action == 0:
        if env.shares_held > 0 and row_now["Close"] > row["Close"] and row_pred["Pred"] > row["Close"]:
            reward_bonus += 0.25
        elif env.shares_held > 0 and row_pred["Pred"] < row["Close"] and price < row["Close"]:
            reward_bonus -= 0.2

    # === Recompensa base por ganancia de capital
    reward += np.tanh(capital_gain / 20)

    if capital_gain > 0:
        reward += 0.5 + np.tanh(capital_gain / 10)
    else:
        reward -= 1 + 0.003 * -capital_gain

    if env.shares_held > 0 and price < env.avg_buy_price:
        reward -= 0.005 * env.shares_held

    if action in [1, 2] and abs(price - env.avg_buy_price) < 0.5:
        reward -= 0.05

    if env.balance < 1:
        reward -= 0.1

    env.steps_without_action += 1
    if env.steps_without_action > 20:
        reward -= 0.05

    if capital_gain < 0 and reward > 0:
        reward -= 0.1

    if action in [1, 2] and reward > 0.5:
        reward += 0.1

    if env.shares_held > 10 and price < env.avg_buy_price:
        reward -= 0.05 * (env.shares_held / 10)

    reward += reward_bonus
    return reward
