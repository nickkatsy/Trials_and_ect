class BertrandFirm:
    def __init__(self, price, cost):
        self.price = price
        self.cost = cost

    def set_price(self, competitor_price):
        self.price = max(0, competitor_price - 3.99)

    def get_profit(self, demand):
        return max(0, (self.price - self.cost) * demand)

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

demand = 50
firm1 = BertrandFirm(price=4, cost=4)
firm2 = BertrandFirm(price=10, cost=10)
market = Market(demand=demand)
market.simulate_bertrand_competition(firm1, firm2)
