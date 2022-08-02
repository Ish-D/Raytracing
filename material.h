#ifndef MATERIAL_H
#define MATERIAL_H

#include "rtweekend.h"
// #include "pdf.h"
#include "texture.h"
#include "curand_kernel.h"

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
  vec3 p;
  do {
    p = 2.0f*RANDVEC3 - vec3(1,1,1);
  } while (p.length_squared() >= 1.0f);
  return p;
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
    
    if (discriminant > 0) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

struct hit_record;

class material {
    public:

        __device__ virtual color emitted(const ray& r_in, const hit_record& rec, float u, float v, const point3& p) const {
            return color();
        }

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const {
            return false;
        }
};

class lambertian : public material {
    public:
        // __device__ lambertian(color c) : albedo(c) {}
        __device__ lambertian(Texture *a) : albedo(a) {}

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const {

            vec3 scatter_dir = rec.normal + random_in_unit_sphere(local_rand_state);

            if (scatter_dir.near_zero()) scatter_dir = rec.normal;

            scattered = ray(rec.p, scatter_dir, r_in.time());
            attenuation = albedo->value(rec.u, rec.v, rec.p);

            return true;
        }
        
    public:
        Texture *albedo;
};


class metallic : public material {
    public:
        __device__ metallic(const color& a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}

        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const  {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(local_rand_state), r_in.time());
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0.0f);
        }

    public:
        color albedo;
        float fuzz;
};

class dielectric : public material {
    public:
        __device__ dielectric(float index_of_refraction) : ir(index_of_refraction) {}

        __device__ virtual bool scatter(
            const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const override {
            vec3 outward_normal;
            vec3 reflected = reflect(r_in.direction(), rec.normal);
            float ni_over_nt;
            attenuation = vec3(1.0, 1.0, 1.0);
            vec3 refracted;
            float reflect_prob;
            float cosine;
            if (dot(r_in.direction(), rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ir;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = sqrt(1.0f - ir*ir*(1-cosine*cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ir;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = reflectance(cosine, ir);
        else
            reflect_prob = 1.0f;
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = ray(rec.p, reflected, r_in.time());
        else
            scattered = ray(rec.p, refracted, r_in.time());
        return true;
        }

    public:
        float ir; // Index of Refraction

    private:
        __device__ static float reflectance(float cosine, float ref_idx) {
            // Use Schlick's approximation for reflectance.
            auto r0 = (1-ref_idx) / (1+ref_idx);
            r0 = r0*r0;
            return r0 + (1-r0)*pow((1 - cosine),5);
        }
};

class diffuse_light : public material {
    public:
        __device__ diffuse_light(Texture *a) : emit(a) {}
        // __device__ diffuse_light(color c) : emit(c) {}

        __device__ virtual color emitted(const ray& r_in, const hit_record& rec, float u, float v, const point3& p) const override {
            // if (!rec.front_face)
            //     return color(0,0,0);
            return emit->value(u, v, p);
        }

    public:
        Texture *emit;
};

#endif