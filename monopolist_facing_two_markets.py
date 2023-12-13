import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


MC1 = 5
MC2 = 3


def demand_function1(price):
    return 20 - price

def demand_function2(price):
    return 15 - 0.5 * price


def total_profit(prices):
    P1, P2 = prices
    Q1 = demand_function1(P1)
    Q2 = demand_function2(P2)
    profit = (P1 - MC1) * Q1 + (P2 - MC2) * Q2
    return -profit  # Negative because we are maximizing


result = minimize(fun=lambda prices: total_profit(prices), x0=[10,10], method='BFGS')

optimal_prices = result.x
max_total_profit = -result.fun


P1_range = np.linspace(0,20,100)
P2_range = np.linspace(0,20,100)
P1_values, P2_values = np.meshgrid(P1_range, P2_range)
profit_values = total_profit([P1_values, P2_values])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(P1_values, P2_values, profit_values, cmap='viridis', alpha=0.7)
ax.scatter(optimal_prices[0], optimal_prices[1], max_total_profit, color='red', s=100, label='Optimal Point')
ax.set_xlabel('Price in Market 1')
ax.set_ylabel('Price in Market 2')
ax.set_zlabel('Total Profit')
ax.set_title('Optimization: Maximize Total Profit with respect to Prices in Two Markets')
ax.legend()

plt.show()

print(f"The optimal prices for Market 1 and Market 2 for maximum total profit are: {optimal_prices}")
print(f"The maximum total profit is: {max_total_profit}")
