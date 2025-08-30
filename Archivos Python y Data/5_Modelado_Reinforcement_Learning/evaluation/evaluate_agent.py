import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# === CONFIGURACIÓN ===
TICKER_MODEL = "GC=F"
TICKER_DATA = "GC=F"
N_STACK = 10
USE_SMALL_INVESTOR = True

# === Rutas ===
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "results", "models", f"{TICKER_MODEL}_dqn_model.zip") # MODIFICAR RESPECTO AL MODELO QUE SE QUIERA ENTRENAR
DATA_PATH = os.path.join(ROOT_DIR, "data", "rl_input", f"{TICKER_DATA}_rl_input.csv")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots", f"{TICKER_MODEL}_on_{TICKER_DATA}")
os.makedirs(PLOT_DIR, exist_ok=True)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# === Cargar datos y entorno ===
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

from env.stock_trading_env import StockTradingEnv
from rewards.reward_fn_tsla import reward_fn_tsla


def make_env():
    return StockTradingEnv(
        df,
        transaction_cost=0.002,
        max_shares_per_trade=3,
        reward_fn=reward_fn_tsla
    )


env = DummyVecEnv([make_env])
#env = VecFrameStack(env, n_stack=N_STACK)
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
        "avg_buy_price": env_.avg_buy_price,
        "total_asset": capital,
        "reward": reward,
        "real_shares_bought": env_.last_shares_bought,
        "real_shares_sold": env_.last_shares_sold
    })

    if done:
        break

# === Guardar CSV con actividad real
csv_path = os.path.join(RESULTS_DIR, f"{TICKER_MODEL}_on_{TICKER_DATA}_portfolio_tracking.csv")
portfolio_df = pd.DataFrame(portfolio_log)
portfolio_df.to_csv(csv_path, index=False)

# === Métricas iniciales
total_real_buys = (portfolio_df["real_shares_bought"] > 0).sum()
total_real_sells = (portfolio_df["real_shares_sold"] > 0).sum()
sell_opportunities = portfolio_df[
    (portfolio_df["shares_held"] > 0) &
    (portfolio_df["price"] > portfolio_df["avg_buy_price"]) &
    (portfolio_df["real_shares_sold"] == 0)
]
missed_sells = len(sell_opportunities)

print(f"\n📌 Acciones realmente compradas: {total_real_buys}")
print(f"📌 Acciones realmente vendidas  : {total_real_sells}")
print(f"\n🚨 Oportunidades de venta no aprovechadas: {missed_sells}")

# === Gráficos ===
portfolio_df["delta_asset"] = portfolio_df["total_asset"].diff()

buy_idx = [i for i, a in enumerate(actions_taken) if a == 1]
sell_idx = [i for i, a in enumerate(actions_taken) if a == 2]
hold_idx = [i for i, a in enumerate(actions_taken) if a == 0]

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(14, 20), gridspec_kw={'height_ratios': [2, 1, 1, 1]})

axes[0].plot(capital_history, label="Capital del Agente", linewidth=2)
axes[0].scatter(buy_idx, [capital_history[i] for i in buy_idx], marker='^', color='green', label='Buy', zorder=5)
axes[0].scatter(sell_idx, [capital_history[i] for i in sell_idx], marker='v', color='red', label='Sell', zorder=5)
axes[0].scatter(hold_idx, [capital_history[i] for i in hold_idx], marker='.', color='gray', label='Hold', zorder=3)
axes[0].set_title(f"Capital usando modelo {TICKER_MODEL} sobre datos de {TICKER_DATA}")
axes[0].set_ylabel("Total Asset ($)")
axes[0].legend()
axes[0].grid(True)


axes[1].plot(dates, np.cumsum(rewards), color="green")
axes[1].set_title("Reward acumulado paso a paso")
axes[1].set_ylabel("Reward")
axes[1].set_xlabel("Fecha")
axes[1].grid(True)

axes[2].plot(portfolio_df["total_asset"], color="blue", label="Capital")
axes[2].scatter(sell_opportunities.index, sell_opportunities["total_asset"],
               color="orange", label="Venta no ejecutada", marker="x", s=30)
axes[2].set_title("Capital del Agente con Oportunidades de Venta no Aprovechadas")
axes[2].set_xlabel("Paso de tiempo (día)")
axes[2].set_ylabel("Total Asset ($)")
axes[2].legend()

axes[3].scatter(portfolio_df["reward"], portfolio_df["delta_asset"], alpha=0.4, s=20, color="blue")
axes[3].axhline(0, color="gray", linestyle="--")
axes[3].axvline(0, color="gray", linestyle="--")
axes[3].set_title("Relación entre Reward y Δ Capital")
axes[3].set_xlabel("Reward obtenido en t")
axes[3].set_ylabel("Δ Capital total")
axes[3].grid(True)

# Guardar gráfico
plot_file = f"{TICKER_MODEL}_on_{TICKER_DATA}_panel_unificado.png"
plt.savefig(os.path.join(PLOT_DIR, plot_file))
plt.close()

print(f"\n[📊] Gráfico resumen guardado en: {os.path.join(PLOT_DIR, plot_file)}")

# === Métricas finales
action_labels = ["Hold", "Buy", "Sell"]
action_counts = [actions_taken.count(0), actions_taken.count(1), actions_taken.count(2)]

duracion = len(capital_history)
ganancia_neta = capital_history[-1] - env_.initial_balance
retorno_pct = (capital_history[-1] / env_.initial_balance - 1) * 100
porcentaje_buy = 100 * action_counts[1] / duracion
porcentaje_sell = 100 * action_counts[2] / duracion
porcentaje_hold = 100 * action_counts[0] / duracion
acciones_mantenidas_prom = np.mean([p["shares_held"] for p in portfolio_log])
tiempo_sin_cash = sum([1 for p in portfolio_log if p["balance"] < 1])
pct_sin_liquidez = 100 * tiempo_sin_cash / duracion

capital_arr = np.array(capital_history)
max_capital = np.max(capital_arr)
min_capital = np.min(capital_arr)
max_drawdown = max_capital - min_capital

# === Imprimir resumen
print(f"\n📊 Evaluación RL - Modelo {TICKER_MODEL} sobre {TICKER_DATA}")
print(f"Capital inicial          : ${env_.initial_balance:.2f}")
print(f"Capital final            : ${capital_history[-1]:.2f}")
print(f"Retorno porcentual       : {retorno_pct:.2f}%")
print(f"Retorno Buy & Hold       : {((buy_and_hold[-1]/env_.initial_balance - 1)*100):.2f}%")
print(f"Acciones compradas       : {action_counts[1]}")
print(f"Acciones vendidas        : {action_counts[2]}")
print(f"Veces que mantuvo (hold) : {action_counts[0]}")
print(f"Promedio de acciones mantenidas: {acciones_mantenidas_prom:.2f}")
print(f"Tiempo sin liquidez (cash < $1): {pct_sin_liquidez:.2f}%")
print(f"Máximo capital alcanzado       : ${max_capital:.2f}")
print(f"Mínimo capital alcanzado       : ${min_capital:.2f}")
print(f"Máximo drawdown                : ${max_drawdown:.2f}")
