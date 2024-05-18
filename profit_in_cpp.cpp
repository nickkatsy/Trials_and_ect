#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>


double f_profit(double x) {
    if (x < 0) {
        return 0;
    }
    if (x == 0) {

        return -std::numeric_limits<double>::infinity();
    }
    double y = exp(-2 * x);
    return 4 * pow(x, 2) * y;
}

int main() {

    double xmin = -1.0;
    double xmax = 500.0;
    int num_points = 500;


    std::vector<double> xv(num_points);
    double step = (xmax - xmin) / (num_points - 1);
    for (int i = 0; i < num_points; ++i) {
        xv[i] = xmin + i * step;
    }

    std::vector<double> fx(num_points);
    std::transform(xv.begin(), xv.end(), fx.begin(), f_profit);


    std::cout << "x\tf(x)" << std::endl;
    for (int i = 0; i < num_points; ++i) {
        std::cout << xv[i] << "\t" << fx[i] << std::endl;
    }

    return 0;
}