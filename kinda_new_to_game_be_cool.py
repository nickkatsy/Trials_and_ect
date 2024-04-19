import numpy as np
import matplotlib.pyplot as plt

def local_roads_benefit(x):
    return 1 + 9*x - 10*x**2

freeway_benefit = 1.8

x_values = np.linspace(0, 1, 100)
y_values_local_roads = local_roads_benefit(x_values)
y_values_freeway = np.full_like(x_values, freeway_benefit)

# part A


def plots(x_values, y_values_freeway, y_values_local_roads):
    plt.figure(figsize=(12,5))
    plt.plot(x_values, y_values_local_roads, label='Benefit of using local roads: $1 + 9x$')
    plt.plot(x_values, y_values_freeway, label=f'Benefit of using freeway: {freeway_benefit}', linestyle='--')

    plt.title('Benefit of Using Roads vs Fraction of Population')
    plt.xlabel('Fraction of population using local roads ($x$)')
    plt.ylabel('Benefit')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plots(x_values, y_values_freeway, y_values_local_roads)


#b

#all equillbrium

from scipy.optimize import fsolve


def possible_equilibriums(x):
    return 1 + 9*x - 10*x**2 - 1.8

equilibrium_x_values = fsolve(possible_equilibriums, [0.1, 0.9])

print("answer:", equilibrium_x_values)
#^^^^ Answer


### I will do this on paper because sympy


###3 however, I will write this out from chapter 10,
# Not very good but I cannot wait to have fun with it.
#;)


def tit_for_tat1(opponent_moves):
    if 'D' in opponent_moves:
        return 'D'
    else:
        return 'C'

def tit_for_tat2(opponent_moves):
    if not opponent_moves:
        return 'D'
    else:
        return opponent_moves[-1]

def play_game_tft(player1_strategy, player2_strategy, num_rounds):
    player1_moves = []
    player2_moves = []
    for _ in range(num_rounds):
        p1_move = player1_strategy(player2_moves)
        p2_move = player2_strategy(player1_moves)
        player1_moves.append(p1_move)
        player2_moves.append(p2_move)
    return player1_moves, player2_moves

player1_moves1, player2_moves1 = play_game_tft(tit_for_tat1, tit_for_tat2, 50)

print("Player 1 (Tit for Tat):", player1_moves1)
print("Player 2 (Tit for Tat):", player2_moves1)


### grim


def grim(opponent_moves):
    if 'D' in opponent_moves:
        return 'D'
    else:
        return 'D'


def grim2(opponent_moves):
    if not opponent_moves:
        return 'D'
    else:
        return opponent_moves[-1]

def play_game_grim(player1_strategy, player2_strategy, num_rounds):
    player1_moves = []
    player2_moves = []
    for _ in range(num_rounds):
        p1_move = player1_strategy(player2_moves)
        p2_move = player2_strategy(player1_moves)
        player1_moves.append(p1_move)
        player2_moves.append(p2_move)
    return player1_moves, player2_moves


player1_moves1, player2_moves1 = play_game_grim(grim, grim2,50)


print("Player 1 (grim):", player1_moves1)
print("Player 2 (grim2):", player2_moves1)












class BertrandFirm:
    def __init__(self, price, cost):
        self.price = price
        self.cost = cost

    def set_price(self, competitor_price):
        self.price = max(0, competitor_price - 0.01)

    def get_profit(self, demand):
        return (self.price - self.cost) * demand

class Market:
    def __init__(self, demand):
        self.demand = demand

    def simulate_bertrand_competition(self, firm1, firm2):
        while True:
            firm1.set_price(firm2.price)
            firm2.set_price(firm1.price)
            total_demand = self.demand - (firm1.price + firm2.price)
            if total_demand < 0:
                total_demand = 0
            firm1_profit = firm1.get_profit(total_demand)
            firm2_profit = firm2.get_profit(total_demand)
            if firm1_profit <= 0 and firm2_profit <= 0:
                break
            print("Firm 1 Price:", firm1.price, "| Firm 1 Profit:", firm1_profit)
            print("Firm 2 Price:", firm2.price, "| Firm 2 Profit:", firm2_profit)
            print("Total Demand:", total_demand)


demand = 450
firm1 = BertrandFirm(price=180, cost=0)
firm2 = BertrandFirm(price=120, cost=0)
market = Market(demand=demand)
market.simulate_bertrand_competition(firm1, firm2)
