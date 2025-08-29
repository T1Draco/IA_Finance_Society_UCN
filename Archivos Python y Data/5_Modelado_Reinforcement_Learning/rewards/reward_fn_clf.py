import numpy as np


def reward_fn_clf(env, action, row, price, capital_gain):
    """
    Función de recompensa específica para trading de oro (GC=F).

    El oro tiene características únicas:
    - Activo refugio en volatilidad
    - Movimientos más suaves que acciones
    - Correlación inversa con USD y mercados en crisis
    - Tendencias de largo plazo más marcadas
    """
    reward = 0
    reward_bonus = 0

    row_now = env.df.iloc[env.current_step]
    row_pred = row_now  # Predicción actual

    # Calcular volatilidad del mercado (usando desviación del SMA)
    volatility = abs(price - row["SMA_20"]) / row["SMA_20"] if row["SMA_20"] > 0 else 0
    is_high_volatility = volatility > 0.02  # 2% de volatilidad

    # Calcular tendencia (pendiente SMA_20 vs SMA_50)
    trend_strength = (row["SMA_20"] - row["SMA_50"]) / row["SMA_50"] if row["SMA_50"] > 0 else 0
    is_uptrend = trend_strength > 0.005  # Tendencia alcista fuerte
    is_downtrend = trend_strength < -0.005  # Tendencia bajista fuerte

    # === Acción: Comprar ===
    if action == 1:
        if env.last_shares_bought > 0:
            # Bonificar comprar en tendencia alcista
            if is_uptrend:
                reward_bonus += 0.3

            # Bonificar comprar en alta volatilidad (oro como refugio)
            if is_high_volatility:
                reward_bonus += 0.25

            # Bonificar comprar por debajo de SMA_20 (comprar en dips)
            if price < row["SMA_20"]:
                reward_bonus += 0.2

            # Bonificar si RSI está sobreventa (<30)
            if row.get("RSI_14", 50) < 30:
                reward_bonus += 0.15

            # Bonificar si MACD es alcista
            if row.get("MACD", 0) > row.get("MACD_signal", 0):
                reward_bonus += 0.1

            # Bonificar si predicción es alcista
            if row_pred["Pred"] > price:
                reward_bonus += 0.2
            elif row_pred["Pred"] <= price * 0.98:  # Predicción muy bajista
                reward -= 0.3

            # Penalizar comprar sin suficiente liquidez de seguridad
            if env.balance < price * env.last_shares_bought * 1.2:
                reward -= 0.25

        else:
            reward -= 0.15  # Penalizar intento fallido de compra

    # === Acción: Vender ===
    elif action == 2:
        if env.last_shares_sold > 0:
            # Gran bonificación por vender con ganancia
            profit_pct = (price - env.avg_buy_price) / env.avg_buy_price if env.avg_buy_price > 0 else 0

            if profit_pct > 0:
                reward_bonus += 1.2  # Bonificación base por ganancia

                # Bonificación escalada por % de ganancia
                if profit_pct > 0.03:  # >3% ganancia
                    reward_bonus += 0.4
                if profit_pct > 0.05:  # >5% ganancia
                    reward_bonus += 0.3
                if profit_pct > 0.08:  # >8% ganancia
                    reward_bonus += 0.2
            else:
                # Penalizar venta con pérdida, pero menos si es stop-loss inteligente
                reward_bonus -= 0.6
                if profit_pct < -0.05:  # Pérdida >5%
                    reward_bonus -= 0.3

            # Bonificar vender en sobrecompra
            if row.get("RSI_14", 50) > 70:
                reward_bonus += 0.2

            # Bonificar vender cuando MACD se vuelve bajista
            if row.get("MACD", 0) < row.get("MACD_signal", 0):
                reward_bonus += 0.15

            # Bonificar vender si predicción es bajista
            if row_pred["Pred"] < price * 0.98:
                reward_bonus += 0.25

            # Bonificar vender en tendencia bajista
            if is_downtrend:
                reward_bonus += 0.2

            # Penalizar vender en tendencia alcista fuerte (salir muy temprano)
            if is_uptrend and profit_pct < 0.02:
                reward -= 0.2

        else:
            reward -= 0.2  # Penalizar intento fallido de venta

    # === Acción: Hold ===
    elif action == 0:
        price_change_pct = (row_now["Close"] - price) / price if price > 0 else 0

        # Penalizar hold cuando hay oportunidades claras
        if abs(price_change_pct) > 0.015:  # Movimiento >1.5%
            if price_change_pct > 0 and env.shares_held == 0:
                reward -= 0.15  # Perdió oportunidad de compra
            elif price_change_pct < 0 and env.shares_held > 0:
                reward -= 0.12  # Perdió oportunidad de venta

        # Penalizar hold con acciones en pérdida significativa
        if env.shares_held > 0 and env.avg_buy_price > 0:
            unrealized_loss_pct = (price - env.avg_buy_price) / env.avg_buy_price
            if unrealized_loss_pct < -0.03:  # Pérdida >3%
                reward -= 0.2

        # Leve penalización base por inacción
        reward -= 0.008

    # === Recompensas generales ===

    # Recompensa base por ganancia de capital (más suave para oro)
    reward += np.tanh(capital_gain / 25)  # Denominador mayor para oro

    # Bonificar crecimiento consistente del portafolio
    if capital_gain > 0:
        reward += 0.08

    # Penalizar pérdidas de capital
    if capital_gain < 0:
        reward -= 0.06
        # Penalización extra por pérdidas grandes
        if capital_gain < -20:
            reward -= 0.15

    # Penalizar quedarse sin liquidez
    if env.balance < 10:
        reward -= 0.25

    # Penalizar trading sin convicción (movimientos muy pequeños)
    if action in [1, 2] and env.avg_buy_price > 0:
        price_diff_pct = abs(price - env.avg_buy_price) / env.avg_buy_price
        if price_diff_pct < 0.005:  # <0.5% diferencia
            reward -= 0.08

    # Bonificar mantener posiciones ganadoras en tendencia alcista
    if env.shares_held > 0 and price > env.avg_buy_price and is_uptrend:
        reward += 0.05

    # Penalizar mantener posiciones perdedoras en tendencia bajista
    if env.shares_held > 0 and price < env.avg_buy_price and is_downtrend:
        reward -= 0.08

    # Bonificar diversificación temporal (no hacer todo-in/todo-out)
    if action == 1 and env.shares_held > 0:  # Compra gradual
        reward += 0.05
    elif action == 2 and env.balance > 50:  # Venta gradual manteniendo liquidez
        reward += 0.05

    reward += reward_bonus
    return reward