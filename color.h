#ifndef COLORPNG_H
#define COLORPNG_H

#include "vec3.h"
#include <iostream>

void write_color(color pixel_color, int samples_per_pixel, uint8_t pixels[], int& i){
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    //Divide color by number of samplesc
    auto scale = 1.0/samples_per_pixel;
    r = sqrt(scale*r);
    g = sqrt(scale*g);
    b = sqrt(scale*b);
    
    // Write the translated value of each color component [0,255]
    pixels[i++] = static_cast<int>(256 * clamp(r, 0.0, 0.999));
    pixels[i++] = static_cast<int>(256 * clamp(g, 0.0, 0.999));
    pixels[i++] = static_cast<int>(256 * clamp(b, 0.0, 0.999));
}

#endif