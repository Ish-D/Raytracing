#ifndef CAMERA_H
#define CAMERA_H

#include "rtweekend.h"
#include "curand_kernel.h"

__device__ vec3 random_in_unit_disk(curandState *local_rand_state) {
  vec3 p;
  do {
    p = 2.0f*vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),0.0f) - vec3(1,1,0);
  } while (dot(p,p) >= 1.0f);
  return p;
}


class camera {
    public:
        __device__ camera(
            point3 lookfrom,
            point3 lookat,
            vec3 vup,
            float vfov, // vertical field of view
            float aspect_ratio,
            float aperture,
            float focus_dist,
            float _time0 = 0,
            float _time1 = 0
        ) {
            auto theta = vfov*(CUDART_PI/180.0f);
            auto h = tan(theta/2);
            auto viewport_height = 2.0*h;
            auto viewport_width = aspect_ratio * viewport_height;

            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = (cross(w, u)); 
            
            origin = lookfrom;
            horizontal = focus_dist * viewport_width *u;
            vertical = focus_dist * viewport_height *v;
            lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;

            lens_radius = aperture/2;
            time0 = _time0;
            time1 = _time1;
        }
        
        __device__ ray get_ray(float s, float t, curandState *local_rand_state) const {
            vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
            vec3 offset = u*rd.x() + v*rd.y();

            float time = time0 + curand_uniform(local_rand_state)*(time1-time0);

            return ray(
                origin+offset, 
                lower_left_corner+ s*horizontal + t*vertical - origin - offset, time);
        }

    private:
        point3 origin;
        point3 lower_left_corner;
        vec3 horizontal;
        vec3 vertical;
        vec3 u,v,w;
        float lens_radius;
        float time0, time1;
};

#endif