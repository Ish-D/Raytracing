#include <iostream>
#include <time.h>
#include "curand_kernel.h"

#include "rtweekend.h"
#include "ray.h"
#include "color.h"
#include "ray.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ color ray_color(const ray& r) {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f*(unit_direction.y() + 1.0f);
    return (1.0f-t)*vec3(1.0,1.0,1.0) + t*vec3(0.5, 0.7, 1.0);
}

__global__ void render (vec3 *fb, int max_x, int max_y, vec3 lower_left_corner, vec3 horizontal, vec3 vertical, vec3 origin) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if ((i >= max_x) || (j >= max_y)) return;

    int pixelIndex = j*max_x + i;
    
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    ray r(origin, lower_left_corner + u*horizontal + v*vertical);
    fb[pixelIndex] = ray_color(r);
}

int main(void) {
    auto aspect_ratio = 1.0/1.0;
    int image_width = 1000;
    int image_height = static_cast<int>(image_width/aspect_ratio);

    int threadsX = 16;
    int threadsY = 16;

    int samples = 50;
    int depth = 50;
    int numPixels = image_width*image_height;

    // Allocate frame buffer
    size_t fb_size = numPixels*sizeof(vec3);
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    dim3 blocks(image_width/threadsX+1, image_height/threadsY+1);
    dim3 threads(threadsX, threadsY);

    clock_t start, stop;
    start = clock();

    // render buffer
    render<<<blocks, threads>>>(fb, image_width, image_height, 
                                    vec3(-2.0, -1.0, -1.0),
                                    vec3(4.0, 0.0, 0.0),
                                    vec3(0.0, 2.0, 0.0),
                                    vec3(0.0, 0.0, 0.0));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Done. Took " << timer_seconds << " seconds.\n";

    // output framebuffer as image
    uint8_t* pixels = new uint8_t[image_height*image_width*3];
    int p = 0;

    for (int j = image_height-1; j>=0; j--) {
        for (int i = 0; i<image_width; i++) {

            size_t pixelIndex = j*image_width + i;
            write_color(fb[pixelIndex], 1, pixels, p);
        }
    }
    stbi_write_png("image.png", image_width, image_height, 3, pixels, image_width*3);
    // std::cerr << "\nDone.\n";
    checkCudaErrors(cudaFree(fb));

    return 0;
}