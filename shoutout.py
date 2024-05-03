from scipy.optimize import fsolve
import numpy as np


def u(n, p):
    return n - p

def buy(p,n):
    if u(p,n) >= 0:
        return 1
    else:
        return 0

def total_demand(p, consumer_types):
    demand_vector = [buy(p, n) / len(consumer_types) for n in consumer_types]
    return sum(demand_vector)

def profit(p1, p2, c1):
    profits = 0.5 * total_demand(p1, consumer_types) * (p1 - c1) if p1 == p2 else total_demand(p1, consumer_types) * (p1 - c1)
    return profits

def reaction(p2, c1):
    return c1 + 0.8 * (p2 - c1) if p2 > c1 else c1

def vector_reaction(p, param):
    return np.array(p) - np.array([reaction(p[1], param[0]), reaction(p[0], param[1])])

def collusion_profits(p, c, delta):
    profits = profit(p, p, c)
    ans = fsolve(vector_reaction, [0.5, 0.5], args=([c, c]))
    if profits >= (1 - delta) * 2 * profits + delta * profit(ans[0], ans[1], c):
        industry_profits = 2 * profits
    else:
        industry_profits = 0
    return industry_profits


consumer_types = np.arange(0.0, 1.01, 0.01)


c = 0.2
delta1 = 0.8
delta2 = 0.4


range_p = np.arange(0.0,1.01,0.01)
range_profits = [collusion_profits(p, c, delta1) for p in range_p]
range_profits_2 = [collusion_profits(p, c, delta2) for p in range_p]
print(range_p)
print(range_profits)
print(range_profits_2)