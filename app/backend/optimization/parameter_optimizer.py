from itertools import product

def optimize_strategy(data, params):
    best_performance = -float('inf')
    best_params = None

    for param in product(*params.values()):
        # Implementa tu lógica de evaluación
        performance = evaluate_strategy(data, param)
        if performance > best_performance:
            best_performance = performance
            best_params = param

    return best_params

# Ejemplo:
params = {'sma_period': [10, 15, 20], 'stop_loss': [0.02, 0.05]}
best_params = optimize_strategy(data, params)
print(best_params)
