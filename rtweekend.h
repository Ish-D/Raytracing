#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <limits>
#include <memory>
#include <cstdlib>
#include "math_constants.h"


// Usings
// using std::shared_ptr;
// using std::make_shared;

// Utility Functions

__device__ inline double degrees_to_rad(double degrees) {
    return degrees * CUDART_PI/180.0;
}

inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

// Common headers
#include "ray.h"
#include "vec3.h"

#endif