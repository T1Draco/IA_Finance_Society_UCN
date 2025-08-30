import numpy as np


def reward_fn_fractional_gcf(env, action_value, row, price, capital_gain):
    """
    Función de recompensa para trading fraccionario de oro (GC=F).

    Args:
        env: Environment instance
        action_value: Valor continuo de acción [-1, 1]
        row: Fila actual de datos
        price: Precio actual
        capital_gain: Ganancia/pérdida de capital en el paso
    """
    reward = 0
    reward_bonus = 0

    row_now = env.df.iloc[env.current_step]
    row_pred = row_now

    # Calcular métricas técnicas
    volatility = abs(price - row["SMA_20"]) / row["SMA_20"] if row["SMA_20"] > 0 else 0
    is_high_volatility = volatility > 0.02

    trend_strength = (row["SMA_20"] - row["SMA_50"]) / row["SMA_50"] if row["SMA_50"] > 0 else 0
    is_uptrend = trend_strength > 0.005
    is_downtrend = trend_strength < -0.005

    # Clasificar acción
    is_buy_action = action_value > 0.1
    is_sell_action = action_value < -0.1
    is_hold_action = abs(action_value) <= 0.1

    # Intensidad de la acción (qué tan agresiva es)
    action_intensity = abs(action_value)

    # === LÓGICA DE COMPRA ===
    if is_buy_action:
        if env.last_shares_bought > 0:  # Compra exitosa
            base_reward = 0.1 * action_intensity  # Reward base proporcional

            # Bonificaciones por timing
            if is_uptrend:
                reward_bonus += 0.2 * action_intensity

            if is_high_volatility:  # Oro como refugio
                reward_bonus += 0.15 * action_intensity

            if price < row["SMA_20"]:  # Comprar en dips
                reward_bonus += 0.15 * action_intensity

            if row.get("RSI_14", 50) < 30:  # Sobreventa
                reward_bonus += 0.1 * action_intensity

            if row.get("MACD", 0) > row.get("MACD_signal", 0):  # MACD alcista
                reward_bonus += 0.08 * action_intensity

            # Bonificar predicciones alcistas
            if row_pred["Pred"] > price:
                reward_bonus += 0.15 * action_intensity
            elif row_pred["Pred"] <= price * 0.98:
                reward -= 0.2 * action_intensity

            # Bonificar diversificación (no ir all-in de una vez)
            if env.position_pct < 0.8 and action_intensity < 0.9:
                reward_bonus += 0.1

            reward += base_reward

        else:  # Intento de compra fallido
            reward -= 0.03 * action_intensity

    # === LÓGICA DE VENTA ===
    elif is_sell_action:
        if env.last_shares_sold > 0:  # Venta exitosa
            base_reward = 0.1 * action_intensity

            # Gran bonificación por vender con ganancia
            if env.avg_buy_price > 0:
                profit_pct = (price - env.avg_buy_price) / env.avg_buy_price

                if profit_pct > 0:
                    # Bonificación escalada por ganancia y cantidad vendida
                    profit_reward = min(profit_pct * 3, 1.0) * action_intensity
                    reward_bonus += profit_reward

                    if profit_pct > 0.03:  # >3% ganancia
                        reward_bonus += 0.2 * action_intensity
                    if profit_pct > 0.05:  # >5% ganancia
                        reward_bonus += 0.15 * action_intensity
                    if profit_pct > 0.08:  # >8% ganancia
                        reward_bonus += 0.1 * action_intensity
                else:
                    # Penalizar venta con pérdida (pero menos si es stop-loss parcial)
                    loss_penalty = min(abs(profit_pct) * 2, 0.5) * action_intensity
                    if action_intensity < 0.7:  # Venta parcial como stop-loss
                        loss_penalty *= 0.5  # Menos penalización
                    reward_bonus -= loss_penalty

            # Bonificaciones por timing técnico
            if row.get("RSI_14", 50) > 70:  # Sobrecompra
                reward_bonus += 0.15 * action_intensity

            if row.get("MACD", 0) < row.get("MACD_signal", 0):  # MACD bajista
                reward_bonus += 0.1 * action_intensity

            if row_pred["Pred"] < price * 0.98:  # Predicción bajista
                reward_bonus += 0.2 * action_intensity

            if is_downtrend:
                reward_bonus += 0.15 * action_intensity

            # Penalizar vender muy temprano en tendencia alcista fuerte
            if is_uptrend and env.avg_buy_price > 0:
                profit_pct = (price - env.avg_buy_price) / env.avg_buy_price
                if profit_pct < 0.02:  # Vender con <2% ganancia en tendencia alcista
                    reward -= 0.1 * action_intensity

            reward += base_reward

        else:  # Intento de venta fallido
            reward -= 0.03 * action_intensity

    # === LÓGICA DE HOLD ===
    elif is_hold_action:
        # Penalización muy leve por inacción
        reward -= 0.001

        # Verificar si perdió oportunidades obvias
        price_change_pct = (row_now["Close"] - price) / price if price > 0 else 0

        if abs(price_change_pct) > 0.02:  # Movimiento significativo >2%
            if price_change_pct > 0 and env.position_pct < 0.2:
                reward -= 0.05  # Perdió oportunidad de compra
            elif price_change_pct < 0 and env.position_pct > 0.8:
                reward -= 0.05  # Perdió oportunidad de venta

        # Penalizar mantener posiciones perdedoras grandes
        if env.shares_held > 0 and env.avg_buy_price > 0:
            unrealized_loss_pct = (price - env.avg_buy_price) / env.avg_buy_price
            if unrealized_loss_pct < -0.05:  # Pérdida >5%
                reward -= 0.1

    # === RECOMPENSAS GENERALES ===

    # Recompensa suavizada por ganancia de capital
    reward += np.tanh(capital_gain / 30)

    # Bonificar crecimiento del portafolio
    if capital_gain > 0:
        reward += 0.05

    # Penalizar pérdidas significativas
    if capital_gain < -10:
        reward -= 0.1

    # Penalizar quedarse sin liquidez para diversificar
    if env.balance < env.min_trade_amount:
        reward -= 0.1

    # Bonificar mantener posiciones ganadoras en tendencia alcista
    if env.shares_held > 0 and env.avg_buy_price > 0 and price > env.avg_buy_price and is_uptrend:
        unrealized_gain = (price - env.avg_buy_price) / env.avg_buy_price
        reward += min(unrealized_gain, 0.1)

    # Penalizar concentración excesiva sin diversificación temporal
    if env.position_pct > 0.95:
        reward -= 0.05
    elif env.position_pct > 0.8 and is_downtrend:
        reward -= 0.08

    # Bonificar gestión de riesgo (posiciones intermedias)
    if 0.3 <= env.position_pct <= 0.7:
        reward += 0.02

    reward += reward_bonus
    return reward