import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# === CONFIGURACIÓN ===
TICKER = "AAPL"
N_STACK = 10
USE_SMALL_INVESTOR = True

# === Rutas ===
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "results", "models" ,f"{TICKER}_dqn_model.zip")
DATA_PATH = os.path.join(ROOT_DIR, "data", "rl_input", f"{TICKER}_rl_input.csv")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots", TICKER)
os.makedirs(PLOT_DIR, exist_ok=True)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# === Cargar datos y entorno ===
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

from env.small_investor_env import SmallInvestorEnv
from env.stock_trading_env import StockTradingEnv

def make_env():
    return SmallInvestorEnv(df) if USE_SMALL_INVESTOR else StockTradingEnv(df)

env = DummyVecEnv([make_env])
env = VecFrameStack(env, n_stack=N_STACK)
model = DQN.load(MODEL_PATH, env=env)

# === Simulación ===
obs = env.reset()
done = False
env_ = env.envs[0]

dates = df["Date"].tolist()[N_STACK-1:]
initial_close = df.loc[N_STACK-1, "Close"]

actions_taken = []
capital_history = []
rewards = []
buy_and_hold = []
portfolio_log = []

for i in range(len(dates)):
    env_.last_shares_bought = 0
    env_.last_shares_sold = 0

    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)

    price = df.loc[i + N_STACK - 1, "Close"]
    capital = env_.balance + env_.shares_held * price
    bh_value = env_.initial_balance * price / initial_close

    capital_history.append(capital)
    buy_and_hold.append(bh_value)
    rewards.append(float(reward))
    actions_taken.append(int(action))

    portfolio_log.append({
        "step": i,
        "date": dates[i],
        "action": int(action),
        "balance": env_.balance,
        "shares_held": env_.shares_held,
        "price": price,
        "total_asset": capital,
        "reward": reward,
        "real_shares_bought": env_.last_shares_bought,
        "real_shares_sold": env_.last_shares_sold
    })

    if done:
        break

# === Guardar CSV con actividad real
portfolio_df = pd.DataFrame(portfolio_log)
portfolio_df.to_csv(os.path.join(RESULTS_DIR, f"{TICKER}_portfolio_tracking.csv"), index=False)

# === NUEVAS métricas
total_real_buys = (portfolio_df["real_shares_bought"] > 0).sum()
total_real_sells = (portfolio_df["real_shares_sold"] > 0).sum()

print(f"\n📌 Acciones realmente compradas: {total_real_buys}")
print(f"📌 Acciones realmente vendidas  : {total_real_sells}")

# === Gráfico 1: Capital + Acciones
plt.figure(figsize=(14, 6))
plt.plot(capital_history, label="Capital del Agente", linewidth=2)

buy_idx = [i for i, a in enumerate(actions_taken) if a == 1]
sell_idx = [i for i, a in enumerate(actions_taken) if a == 2]
hold_idx = [i for i, a in enumerate(actions_taken) if a == 0]

plt.scatter(buy_idx, [capital_history[i] for i in buy_idx], marker='^', color='green', label='Buy', zorder=5)
plt.scatter(sell_idx, [capital_history[i] for i in sell_idx], marker='v', color='red', label='Sell', zorder=5)
plt.scatter(hold_idx, [capital_history[i] for i in hold_idx], marker='.', color='gray', label='Hold', zorder=3)

plt.title(f"Evolución del Capital y Acciones - {TICKER}")
plt.xlabel("Paso de tiempo (día)")
plt.ylabel("Total Asset ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{TICKER}_capital_actions_plot.png"))
plt.close()

# === Gráfico 2: RL vs Buy & Hold
plt.figure(figsize=(14, 5))
plt.plot(dates, capital_history, label="Agente RL", color="blue")
plt.plot(dates, buy_and_hold, label="Buy & Hold", color="orange", linestyle="--")
plt.title(f"Capital Total - RL vs Buy & Hold ({TICKER})")
plt.xlabel("Fecha")
plt.ylabel("Capital ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{TICKER}_rl_vs_buyhold.png"))
plt.close()

# === Gráfico 3: Reward acumulado
plt.figure(figsize=(14, 4))
plt.plot(dates, np.cumsum(rewards), color="green")
plt.title("Reward acumulado paso a paso")
plt.xlabel("Fecha")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{TICKER}_reward_acumulado.png"))
plt.close()

# === Gráfico 4: Distribución de acciones
plt.figure(figsize=(6, 4))
action_labels = ["Hold", "Buy", "Sell"]
action_counts = [actions_taken.count(0), actions_taken.count(1), actions_taken.count(2)]
plt.bar(action_labels, action_counts, color=["gray", "green", "red"])
plt.title("Distribución de Acciones")
plt.ylabel("Cantidad")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{TICKER}_distribucion_acciones.png"))
plt.close()

# === Resumen
print(f"\n📊 Evaluación final del agente RL ({TICKER})")
print(f"Capital inicial          : ${env_.initial_balance:.2f}")
print(f"Capital final            : ${capital_history[-1]:.2f}")
print(f"Retorno porcentual       : {((capital_history[-1]/env_.initial_balance - 1)*100):.2f}%")
print(f"Retorno Buy & Hold       : {((buy_and_hold[-1]/env_.initial_balance - 1)*100):.2f}%")
print(f"Acciones compradas       : {action_counts[1]}")
print(f"Acciones vendidas        : {action_counts[2]}")
print(f"Veces que mantuvo (hold) : {action_counts[0]}")
print(f"[✔] Gráficos y CSV guardados en: {PLOT_DIR}")

# === Métricas adicionales ===
duracion = len(capital_history)
ganancia_neta = capital_history[-1] - env_.initial_balance
retorno_pct = (capital_history[-1] / env_.initial_balance - 1) * 100

# Distribución
porcentaje_buy = 100 * action_counts[1] / duracion
porcentaje_sell = 100 * action_counts[2] / duracion
porcentaje_hold = 100 * action_counts[0] / duracion

# Acciones mantenidas promedio
acciones_mantenidas_prom = np.mean([p["shares_held"] for p in portfolio_log])

# Tiempo sin liquidez
tiempo_sin_cash = sum([1 for p in portfolio_log if p["balance"] < 1])
pct_sin_liquidez = 100 * tiempo_sin_cash / duracion

# Máximos y drawdown
capital_arr = np.array(capital_history)
max_capital = np.max(capital_arr)
min_capital = np.min(capital_arr)
max_drawdown = max_capital - min_capital

# === Imprimir métricas
print(f"\n📊 Métricas del agente RL para {TICKER}")
print(f"Duración: {duracion} pasos")
print(f"Ganancia neta: ${ganancia_neta:.2f}")
print(f"Retorno porcentual: {retorno_pct:.2f}%")
print("\nDistribución de acciones:")
print(f" - Buy:  {porcentaje_buy:.2f}%")
print(f" - Sell: {porcentaje_sell:.2f}%")
print(f" - Hold: {porcentaje_hold:.2f}%")
print(f"\nPromedio de acciones mantenidas: {acciones_mantenidas_prom:.2f}")
print(f"Porcentaje del tiempo sin liquidez: {pct_sin_liquidez:.2f}%")
print(f"Máximo capital alcanzado: ${max_capital:.2f}")
print(f"Mínimo capital alcanzado: ${min_capital:.2f}")
print(f"Máximo drawdown: ${max_drawdown:.2f}")
