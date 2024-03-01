#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <functional>
#include <cassert>
#include <limits>
#include <cmath>
#include <algorithm>
#include <tuple>

double demand(double price) {
    return 300 - price;
}

double cournot_reaction_firm1(double price, double quantity_firm1, double quantity_firm2) {
    return demand(price) - quantity_firm1;
}

double cournot_reaction_firm2(double price, double quantity_firm1, double quantity_firm2) {
    return demand(price) - quantity_firm2;
}

double cournot_objective(const std::vector<double>& prices_quantities) {
    double price = prices_quantities[0];
    double quantity_firm1 = prices_quantities[1];
    double quantity_firm2 = prices_quantities[2];

    double revenue_firm1 = price * cournot_reaction_firm1(price, quantity_firm1, quantity_firm2);
    double revenue_firm2 = price * cournot_reaction_firm2(price, quantity_firm1, quantity_firm2);
    double total_revenue = revenue_firm1 + revenue_firm2;
    return -total_revenue;
}

struct Constraint {
    std::function<double(const std::vector<double>&)> fun;
};

std::vector<Constraint> constraints = {
    {[](const std::vector<double>& x) { return x[0]; }},  // Price >= 0
    {[](const std::vector<double>& x) { return x[1]; }},  // Quantity Firm 1 >= 0
    {[](const std::vector<double>& x) { return x[2]; }}   // Quantity Firm 2 >= 0
};

std::vector<double> initial_guess = {100, 50, 50}; 

std::vector<double> minimize(const std::function<double(const std::vector<double>&)>& objective,
                              const std::vector<double>& initial_guess,
                              const std::vector<Constraint>& constraints) {
    return initial_guess;
}

int main() {
    std::vector<double> result = minimize(cournot_objective, initial_guess, constraints);

    double equilibrium_price = result[0];
    double equilibrium_quantity_firm1 = result[1];
    double equilibrium_quantity_firm2 = result[2];

    std::cout << "Bernant Equilibrium:" << std::endl;
    std::cout << "Price: " << equilibrium_price << std::endl;
    std::cout << "Quantity Firm 1: " << equilibrium_quantity_firm1 << std::endl;
    std::cout << "Quantity Firm 2: " << equilibrium_quantity_firm2 << std::endl;

    return 0;
}