#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <limits>
#include <memory>
#include <cstdlib>

// Usings
using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// Constants
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

inline double degrees_to_rad(double degrees) {
    return degrees * pi/180.0;
}

inline double random_double(){
    // Return a random number [0,1]
    return rand()/(RAND_MAX + 1.0);
}

inline double random_double(double min, double max) {
    // Return a random number [min, max]
    return min + (max-min)*random_double();
}

inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

inline int random_int(int min, int max) {
    // Returns a random integer in [min,max]
    return static_cast<int>(random_double(min, max+1));
}

// Common headers
#include "ray.h"
#include "vec3.h"

#endif