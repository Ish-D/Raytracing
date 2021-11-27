#include "rtweekend.h"

#include <iostream>
#include <iomanip>
#include <math.h>
#include <stdlib.h>


int main() {
    int inside_circle = 0;
    int inside_circle_stratified = 0;
    int sqrt_n = 10000;

    for (int i = 0; i<sqrt_n; i++) {
        for (int j = 0; j<sqrt_n; j++) {
            auto x = random_double(-1, 1);
            auto y = random_double(-1, 1);
            if (x*x + y*y < 1) 
                inside_circle++;
            
            x = 2*((i + random_double())/sqrt_n) -1;   
            y = 2*((j + random_double())/sqrt_n) -1;   
            if (x*x + y*y < 1) 
                inside_circle_stratified++;
        }

    }
    auto N = static_cast<double>(sqrt_n) * sqrt_n;
    std::cout << std::fixed << std::setprecision(12);
    std::cout
        << "Regular    Estimate of Pi = "
        << 4*double(inside_circle) / (sqrt_n*sqrt_n) << '\n'
        << "Stratified Estimate of Pi = "
        << 4*double(inside_circle_stratified) / (sqrt_n*sqrt_n) << '\n';

    return 0;
}