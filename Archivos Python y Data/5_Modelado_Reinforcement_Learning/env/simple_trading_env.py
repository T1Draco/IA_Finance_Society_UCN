import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SimpleTradingEnv(gym.Env):
    """
    Entorno de trading súper simple para empezar:
    - Solo 3 acciones: HOLD, BUY, SELL
    - Recompensa simple: solo ganancia/pérdida
    - Estado mínimo: precio actual, balance, acciones
    """

    def __init__(self, data, initial_balance=1000, transaction_cost=0.001):
        super().__init__()

        # Datos
        self.data = data.copy()
        self.prices = data['Close'].values
        self.n_steps = len(self.prices)

        # Parámetros
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost

        # Espacios
        # Acción: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # Estado: [precio_normalizado, balance_normalizado, shares_normalizadas, profit_pct]
        self.observation_space = spaces.Box(
            low=-5, high=5, shape=(4,), dtype=np.float32
        )

        # Reset inicial
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Estado inicial
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_trades = 0
        self.buy_price = 0  # Precio promedio de compra

        # Historial
        self.history = []

        return self._get_observation(), {}

    def _get_observation(self):
        """Estado simple y normalizado"""
        current_price = self.prices[self.current_step]

        # Normalizar valores
        price_norm = (current_price - self.prices[:self.current_step + 1].mean()) / (
                    self.prices[:self.current_step + 1].std() + 1e-8)
        balance_norm = (self.balance - self.initial_balance) / self.initial_balance
        shares_norm = self.shares_held / 10  # Asumir máximo 10 shares

        # Profit porcentual actual
        portfolio_value = self.balance + self.shares_held * current_price
        profit_pct = (portfolio_value - self.initial_balance) / self.initial_balance

        return np.array([price_norm, balance_norm, shares_norm, profit_pct], dtype=np.float32)

    def step(self, action):
        current_price = self.prices[self.current_step]
        previous_portfolio = self.balance + self.shares_held * current_price

        # Ejecutar acción
        reward = 0
        info = {'action_executed': False}

        if action == 1:  # BUY
            if self.balance >= current_price:
                # Comprar 1 share (simple)
                cost = current_price * (1 + self.transaction_cost)
                if cost <= self.balance:
                    self.balance -= cost
                    self.shares_held += 1
                    self.buy_price = current_price  # Simplificado: último precio de compra
                    self.total_trades += 1
                    info['action_executed'] = True

        elif action == 2:  # SELL
            if self.shares_held > 0:
                # Vender 1 share
                revenue = current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.shares_held -= 1
                self.total_trades += 1
                info['action_executed'] = True

                # Recompensa por ganancia/pérdida
                if self.buy_price > 0:
                    profit = revenue - self.buy_price
                    reward += np.tanh(profit / (current_price * 0.1))  # Recompensa escalada

        # Avanzar tiempo
        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1

        # Calcular recompensa base: cambio en valor del portfolio
        new_portfolio = self.balance + self.shares_held * current_price
        portfolio_change = new_portfolio - previous_portfolio
        reward += np.tanh(portfolio_change / (self.initial_balance * 0.01))

        # Penalizar inacción excesiva
        if action == 0:
            reward -= 0.001

        # Guardar historial
        self.history.append({
            'step': self.current_step - 1,
            'price': current_price,
            'action': action,
            'balance': self.balance,
            'shares': self.shares_held,
            'portfolio_value': new_portfolio,
            'reward': reward
        })

        return self._get_observation(), reward, terminated, False, info

    def render(self):
        """Mostrar estado actual"""
        current_price = self.prices[self.current_step]
        portfolio_value = self.balance + self.shares_held * current_price

        print(f"Step {self.current_step}: Price=${current_price:.2f}, "
              f"Balance=${self.balance:.2f}, Shares={self.shares_held}, "
              f"Portfolio=${portfolio_value:.2f}")


# Función para generar datos de prueba
def create_test_data(n_days=100, start_price=100):
    """Crear datos sintéticos para pruebas"""
    np.random.seed(42)

    # Precio con tendencia y ruido
    trend = np.linspace(0, 0.2, n_days)  # Tendencia alcista leve
    noise = np.random.normal(0, 0.02, n_days)
    returns = trend + noise

    prices = [start_price]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))

    data = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=n_days),
        'Close': prices
    })

    return data


# Función de entrenamiento simple
def train_simple_agent(env, n_episodes=1000):
    """Entrenamiento con estrategia simple (no RL todavía)"""
    from collections import deque

    results = []
    recent_rewards = deque(maxlen=100)

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Estrategia simple basada en estado
            price_trend = obs[0]  # Tendencia del precio
            profit_pct = obs[3]  # Profit actual

            if price_trend < -0.5 and env.shares_held == 0:
                action = 1  # BUY en caídas
            elif price_trend > 0.5 and env.shares_held > 0:
                action = 2  # SELL en subidas
            elif profit_pct < -0.05 and env.shares_held > 0:
                action = 2  # Stop loss
            elif profit_pct > 0.1 and env.shares_held > 0:
                action = 2  # Take profit
            else:
                action = 0  # HOLD

            obs, reward, done, _, info = env.step(action)
            total_reward += reward

        recent_rewards.append(total_reward)

        if episode % 100 == 0:
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode}: Avg Reward = {avg_reward:.3f}")

        results.append({
            'episode': episode,
            'total_reward': total_reward,
            'final_portfolio': env.balance + env.shares_held * env.prices[-1],
            'total_trades': env.total_trades
        })

    return results


if __name__ == "__main__":
    # Crear datos de prueba
    data = create_test_data(252)  # 1 año de trading

    # Crear entorno
    env = SimpleTradingEnv(data, initial_balance=1000)

    print("🎯 Entorno de trading simple creado")
    print(f"Datos: {len(data)} días")
    print(f"Precio inicial: ${data['Close'].iloc[0]:.2f}")
    print(f"Precio final: ${data['Close'].iloc[-1]:.2f}")
    print(f"Buy & Hold return: {((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100:.1f}%")

    # Entrenar agente simple
    print("\n🤖 Entrenando agente con estrategia simple...")
    results = train_simple_agent(env, n_episodes=500)

    # Mostrar resultados
    final_results = pd.DataFrame(results)
    best_episode = final_results.loc[final_results['total_reward'].idxmax()]

    print(f"\n📊 Mejores resultados:")
    print(f"Mejor reward: {best_episode['total_reward']:.3f}")
    print(f"Portfolio final: ${best_episode['final_portfolio']:.2f}")
    print(f"Return: {((best_episode['final_portfolio'] / 1000) - 1) * 100:.1f}%")
    print(f"Trades totales: {best_episode['total_trades']}")

    # Gráfico simple
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(data['Close'])
    plt.title('Precio del Activo')
    plt.ylabel('Precio')

    plt.subplot(1, 2, 2)
    plt.plot(final_results['total_reward'].rolling(50).mean())
    plt.title('Reward Promedio (50 episodios)')
    plt.ylabel('Reward')
    plt.xlabel('Episodio')

    plt.tight_layout()
    plt.show()

    print("\n✅ Prueba completada. Si esto funciona bien, podemos avanzar a RL real.")