#ifndef BOX_H
#define BOX_H

#include "rtweekend.h"
#include "aarect.h"
#include "hittable_list.h"

class box : public hittable {
    public:
        __device__ box() {}
        __device__ box(const point3& p0, const point3& p1, material *ptr);

        __device__ virtual bool hit(const ray& r,  float t_min, float t_max, hit_record& rec) const override;
        
        __device__ virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
            output_box =    aabb(box_min, box_max);
            return true;
        }
    public:
        point3 box_min;
        point3 box_max;
        hittable_list sides;
};

__device__ box::box(const point3& p0, const point3& p1, material *ptr) {
    box_min = p0;
    box_max = p1;

    // sides[hitIndex++] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr);
    // sides[hitIndex++] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr);

    // sides[hitIndex++] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr);
    // sides[hitIndex++] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr);

    // sides[hitIndex++] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr);
    // sides[hitIndex++] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr);

    sides[0] = xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr);
    // sides[1] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr);

    // sides[2] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr);
    // sides[3] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr);

    // sides[4] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr);
    // sides[5] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr);
}

__device__ bool box::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    return sides.hit(r, t_min, t_max, rec);
}

#endif