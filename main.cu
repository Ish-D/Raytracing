#include <iostream>
#include <time.h>
#include "curand_kernel.h"

#include "rtweekend.h"

#include "color.h"
#include "camera.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"
#include "aarect.h"
// #include "box.h"

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define RND (curand_uniform(&local_rand_state))

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


__device__ color ray_color(const ray& r, hittable **world, int depth, curandState *local_rand_state) {
    hit_record rec;

    if ((*world)->hit(r, 0.001f, FLT_MAX, rec)) {
      ray scattered;
      color attenuation;
      color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

      if (depth < 5 && rec.mat_ptr->scatter(r, rec, attenuation, scattered, local_rand_state)) {
        return emitted + attenuation * ray_color(scattered, world, depth+1, local_rand_state);
      }
      else return emitted;
    }

    else {
      return color(0,0,0);
    }
}



__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j*max_x + i;

  //Each thread gets same seed, a different sequence number, no offset
  curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}


__global__ void render (vec3 *fb, int max_x, int max_y, int samples, camera **cam, hittable **world, curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if ((i >= max_x) || (j >= max_y)) return;

  int pixelIndex = j*max_x + i;   

  curandState local_rand_state = rand_state[pixelIndex];
  color col(0,0,0);   

  for (int s = 0; s<samples; s++) {
    float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
    ray r = (*cam)->get_ray(u,v, &local_rand_state);
    col += ray_color(r, world, 0, &local_rand_state);
  }

  col /= float(samples);  
  // col = vec3(0, 100, 0);
  fb[pixelIndex] = col;
  // rand_state[pixelIndex] = local_rand_state;
}

__global__ void create_world(hittable **list, hittable **world, camera **cam, int aspect_ratio, curandState *rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_rand_state = *rand_state;

    auto red = new lambertian(new solid_color(.65, .05, .05));
    auto green = new lambertian(new solid_color(.12, .45, .15));
    auto white = new lambertian(new solid_color(.73, .73, .73));
    auto light = new diffuse_light(new solid_color(100, 100, 100));
    // auto glass = new dielectric(1.5);
    // auto metal = new metallic(color(0.5, 0.5, 0.5), 0.5);

    list[0] = new sphere(point3(400,90,165), 100, green);
    list[1] = new sphere(point3(190, 90, 190), 100, green);

    list[2] = new xz_rect(0, 555, 0, 555, 0, white);
    list[3] = new xz_rect(0, 555, 0, 555, 555, white);
    list[4] = new xy_rect(0, 555, 0, 555, 555, white);
    list[5] = new yz_rect(0, 555, 0, 555, 555, green);
    list[6] = new yz_rect(0, 555, 0, 555, 0, red);
    list[7] = new xz_rect(213, 343, 227, 332, 554, light);

    *rand_state = local_rand_state;
    *world  = new hittable_list(list, 8);

    vec3 vup(0,1,0);
    point3 lookfrom = point3(278, 278, -800);
    point3 lookat = point3(278, 278, 0);
    auto dist_to_focus = (lookfrom-lookat).length();
    auto vfov = 40.0;
    auto aperture = 0.0;
    auto time0 = 0.0;
    auto time1 = 1.0;

    *cam = new camera(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus, time0, time1);
  }
}

__global__ void free_world(hittable **list, hittable **world, camera **camera) {
  for (int i = 0; i<8; i++) {
    delete list[i];
  }

  delete *world;
  delete *camera;
}

int main(void) {
    auto aspect_ratio = 1.0/1.0;
    int image_width = 1000;
    int image_height = static_cast<int>(image_width/aspect_ratio);

    int threadsX = 16;
    int threadsY = 16;

    int samples = 200;
    int numPixels = image_width*image_height;

    // Allocate frame buffer
    size_t fb_size = numPixels*sizeof(vec3);
    vec3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    dim3 blocks(image_width/threadsX+1, image_height/threadsY+1);
    dim3 threads(threadsX, threadsY);

    clock_t start, stop;
    start = clock();

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, numPixels*sizeof(curandState)));
    
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    hittable **list;
    int num_hittables = 8;
    checkCudaErrors(cudaMalloc((void **)&list, num_hittables*sizeof(hittable *)));
    hittable **world;
    checkCudaErrors(cudaMalloc((void **)&world, sizeof(hittable *)));
    camera **cam;
    checkCudaErrors(cudaMalloc((void **)&cam, sizeof(camera *)));
    create_world<<<1,1>>>(list, world, cam, aspect_ratio, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    // render buffer
    render<<<blocks, threads>>>(fb, image_width, image_height, samples, cam, world, d_rand_state);
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
            write_color(fb[pixelIndex], samples, pixels, p);
        }
    }
    stbi_write_png("image.png", image_width, image_height, 3, pixels, image_width*3);
    // std::cerr << "\nDone.\n";
    checkCudaErrors(cudaFree(fb));
    free_world<<<1,1>>>(list, world, cam);
    return 0;
}