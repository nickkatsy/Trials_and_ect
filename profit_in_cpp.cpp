#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

// Profit Function
double f_profit(double x) {
    if (x < 0) {
        return 0;
    }
    if (x == 0) {
        return NAN;
    }
    double y = exp(-2 * x);
    return 4 * pow(x, 2) * y;
}

int main() {
    // Range
    double xmin = -1.0;
    double xmax = 1.0;
    int num_points = 200;

    // Generate x values
    std::vector<double> xv(num_points);
    double step = (xmax - xmin) / (num_points - 1);
    for (int i = 0; i < num_points; ++i) {
        xv[i] = xmin + i * step;
    }

    // Calculate profit values for each x
    std::vector<double> fx(num_points);
    std::transform(xv.begin(), xv.end(), fx.begin(), f_profit);

    // Visuals
    std::cout << "x\tf(x)" << std::endl;
    for (int i = 0; i < num_points; ++i) {
        std::cout << xv[i] << "\t" << fx[i] << std::endl;
    }

    return 0;
}