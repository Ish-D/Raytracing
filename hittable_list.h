#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "rtweekend.h"

#include "hittable.h"

#include <memory>
#include <vector>


class hittable_list : public hittable  {
    public:
        __device__ hittable_list() {}
        __device__ hittable_list(hittable **l, int n) {objects = l; objects_size = n;}
        __device__ hittable_list(int n) {objects_size = n;}
        // __device__ hittable_list(hittable *object) { add(object); }
        // __device__ void clear() { objects.clear(); }
        // __device__ void add(hittable *object) { objects.push_back(object);}

        __device__ virtual bool hit(
            const ray& r, float t_min, float t_max, hit_record& rec) const override;

        __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override;
        // __device__ virtual float pdf_value(const vec3 &o, const vec3 &v) const override;
        // __device__ virtual vec3 random(const vec3 &o) const override;

    public:
        hittable **objects;
        int objects_size;
};


__device__ bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    hit_record temp_rec;
    auto hit_anything = false;
    auto closest_so_far = t_max;

    for (int i = 0; i<objects_size; i++) {
        if (objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}


__device__ bool hittable_list::bounding_box(float time0, float time1, aabb& output_box) const {
    if (objects_size < 1) return false;

    aabb temp_box;
    bool first_box = true;

    for (int i = 0; i<objects_size; i++) {
        if (objects[i]->bounding_box(time0, time1, temp_box)) return false;
        output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
        first_box = false;
    }

    return true;
}


// __device__ float hittable_list::pdf_value(const point3& o, const vec3& v) const {
//     auto weight = 1.0/objects.size();
//     auto sum = 0.0;

//     for (const auto& object : objects)
//         sum += weight * object->pdf_value(o, v);

//     return sum;
// }


// __device__ vec3 hittable_list::random(const vec3 &o) const {
//     auto int_size = static_cast<int>(objects.size());
//     return objects[random_int(0, int_size-1)]->random(o);
// }


#endif