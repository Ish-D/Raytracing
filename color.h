#ifndef COLORPNG_H
#define COLORPNG_H

#include "vec3.h"
#include <iostream>

void write_color(color pixel_color, int samples_per_pixel, uint8_t pixels[], int& i){
    float r = pixel_color.x();
    float g = pixel_color.y();
    float b = pixel_color.z();

    if (r != r) r = 0.0;
    if (g != g) g = 0.0;
    if (b != b) b = 0.0;

    //Divide color by number of samples
    auto scale = 1.0/samples_per_pixel;
    r = sqrt(scale*r);
    g = sqrt(scale*g);
    b = sqrt(scale*b);

    // Write the translated value of each color component [0,255]
    pixels[i++] = static_cast<int>(255.99 * r);
    pixels[i++] = static_cast<int>(256.99 * g);
    pixels[i++] = static_cast<int>(255.99 * b);
}

#endif