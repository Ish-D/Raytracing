#include "rtweekend.h"

#include "camera.h"
#include "color.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include "moving_sphere.h"

//#define STB_IMAGE_IMPLEMENTATION
#include "rtw_stb_image.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <fstream>
#include <sstream>

color ray_color(const ray& r, const hittable& world, int depth) {
    hit_record rec;

    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0)
        return color(0,0,0);

    if (world.hit(r, 0.001, infinity, rec)) {
        ray scattered;
        color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
            return attenuation * ray_color(scattered, world, depth-1);
        return color(0,0,0);
    }
    vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

hittable_list original_scene() {
    
    hittable_list objects;

    auto material_ground = make_shared<checker_texture>(color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9));
    auto material_center = make_shared<lambertian>(color(0.3, 0.3, 1.0));
    auto material_left   = make_shared<dielectric>(1.5);
    auto material_right  = make_shared<metal>(color(0.8, 0.6, 0.2), 0.5);

    objects.add(make_shared<sphere>(point3( 0.0, -100.5, -1.0), 100.0, make_shared<lambertian>(material_ground)));
    objects.add(make_shared<sphere>(point3( 0.0, 0.0, -1.0), 0.5, material_center));
    objects.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.5, material_left));
    objects.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), -0.425, material_left));
    objects.add(make_shared<sphere>(point3( 1.0, 0.0, -1.0), -0.45, material_right));
    objects.add(make_shared<moving_sphere>(point3(0, 0, 0), point3(0,0,-0.5), 0, 1, 0.3, material_center));

    return objects;
}

hittable_list two_spheres() {
    hittable_list objects;

    auto checker = make_shared<checker_texture>(color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9));

    objects.add(make_shared<sphere>(point3(0,-10, 0), 10, make_shared<lambertian>(checker)));
    objects.add(make_shared<sphere>(point3(0, 10, 0), 10, make_shared<lambertian>(checker)));

    return objects;
}

hittable_list perlin_spheres() {

    hittable_list objects; 

    auto pertext = make_shared<noise_texture>(4);
    objects.add(make_shared<sphere>(point3(0, -1000, 0), 1000, make_shared<lambertian>(pertext)));
    objects.add(make_shared<sphere>(point3(0, 2, 0), 2, make_shared<lambertian>(pertext)));

    return objects;
}

hittable_list earth() {
    auto earth_texture = make_shared<image_texture>("mars_texture.jpg");
    auto earth_surface = make_shared<lambertian>(earth_texture);
    auto globe = make_shared<sphere>(point3(0,0,0), 2, earth_surface);

    return hittable_list(globe);
}

int main()
{
    // Canvas.
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 800;
    const int samples_per_pixel = 50;
    const int max_depth = 100;


    // World
    auto r = cos(pi/4);

    hittable_list world;

    point3 lookfrom;
    point3 lookat;
    auto vfov = 40.0;
    auto aperture = 0.0;


    switch(4) {
        case 1:
            world = original_scene();
            lookfrom = point3(3,3,2);
            lookat = point3(0,0,-1);
            vfov = 20.0;
            aperture = 0.5;
            break;

        default:
        case 2:
            world = two_spheres();
            lookfrom = point3(13,2,3);
            lookat = point3(0,0,0);
            vfov = 20.0;
            break;
        case 3:
            world = perlin_spheres();
            lookfrom = point3(13,2,3);
            lookat = point3(0,0,0);
            vfov = 20.0;
            break;
        case 4:
            world = earth();
            lookfrom = point3(13,2,3);
            lookat = point3(0,0,0);
            vfov = 20.0;
            break;
    }

    // Camera
    vec3 vup(0,1,0);
    auto dist_to_focus = (lookfrom-lookat).length();
    int image_height = static_cast<int>(image_width / aspect_ratio);

    camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);

    // Render 
    //std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";

    // Pixels written out from left to right (i = 0)
    // Rows top to bottom (j = image_height-1)
    uint8_t* pixels = new uint8_t[image_height*image_width*3];
    static int pixelIndex = 0;

    for (int j = image_height-1; j>= 0; --j)
    {
        // Progress Bar
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i<image_width; ++i)
        {
            color pixel_color(0,0,0);
            for (int s = 0; s<samples_per_pixel; ++s) {
                auto u = double(i + random_double()) / (image_width-1);
                auto v = double(j + random_double()) / (image_height-1);
                ray r = cam.get_ray(u,v);
                pixel_color += ray_color(r, world, max_depth);
            }
            write_color(pixel_color, samples_per_pixel, pixels, pixelIndex);
        }
    }
    stbi_write_png("image.png", image_width, image_height, 3, pixels, image_width*3);
    std::cerr << "\nDone.\n";
}