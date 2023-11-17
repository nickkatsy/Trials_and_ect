import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


w1 = 1
w2 = 1


def short_term_profit(x1,x2):
    return -(x1**2 + x2**2)


def long_term_profit(x1, x2):
    profit_function = -(x1**2 + x2**2)
    return profit_function - w1*x1 - w2*x2


x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
x1, x2 = np.meshgrid(x1, x2)
short_term_profit_values = short_term_profit(x1,x2)
long_term_profit_values = long_term_profit(x1,x2)


fig = plt.figure(figsize=(12,6))


ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x1, x2, short_term_profit_values, cmap='viridis')
ax1.set_title('Short-term Profit Maximization')
ax1.set_xlabel('X1-axis')
ax1.set_ylabel('X2-axis')
ax1.set_zlabel('Profit')

# lr
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x1, x2, long_term_profit_values, cmap='viridis')
ax2.set_title('Long-term Profit Maximization')
ax2.set_xlabel('X1-axis')
ax2.set_ylabel('X2-axis')
ax2.set_zlabel('Profit')

plt.show()