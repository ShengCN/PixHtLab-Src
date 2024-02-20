#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <stdio.h>

namespace {
    template <typename scalar_t>
    __device__
    scalar_t sign(scalar_t t) {
        if (t > 0.0) {
            return (scalar_t)1.0;
        } else {
            return -(scalar_t)1.0;
        }
    }

    template <typename scalar_t>
    struct vec2 {
        scalar_t x, y;

        __device__
        vec2() { x=0.0, y=0.0;}

        __device__
        vec2(scalar_t x, scalar_t y):x(x), y(y) {}
    };


    template <typename scalar_t>
    struct vec3 {
        scalar_t x, y, z;

        __device__
        vec3() { x=0.0, y=0.0, z=0.0;}

        __device__
        vec3(scalar_t x, scalar_t y, scalar_t z):x(x), y(y), z(z) {}
    };


    template <typename scalar_t>
    __device__
    scalar_t lerp(scalar_t a, scalar_t b, scalar_t t) {
        return (1.0-t) * a + t * b;
    }

    template <typename scalar_t>
    __device__
    void proj_ground(
        scalar_t x0, scalar_t y0, scalar_t h0,
        scalar_t x1, scalar_t y1, scalar_t h1,
        scalar_t &x2, scalar_t &y2
    ) {
        scalar_t t = (0-h0)/(h1-h0);
        x2 = lerp(x0, x1, t);
        y2 = lerp(y0, y1, t);
    }


    // line checking condition with thickness value dh, which is the height difference for double-height map
    // we can also use dh as a tolerance value
    template <typename scalar_t>
    __device__
    bool check_intersect(
        scalar_t xa, scalar_t ya, scalar_t ha,
        scalar_t xb, scalar_t yb, scalar_t hb,
        scalar_t x, scalar_t y, scalar_t h,
        scalar_t dh, int& flag) {
        scalar_t t = xa == xb ? (y-ya)/(yb-ya):(x-xa)/(xb-xa);
        scalar_t h_ = lerp(ha, hb, t);
        flag = h_ <= h ? 1:-1;
        return (h_ <= h) && (h_ >= h-dh);
    }

    /*
    * Ray trace in the current scene
    * Given start point xyh, light point xyh, current receiver's height map,
    * Return:
    *      1. if intersect or not
    *      2. the color for the intersection point
    * */
    template <typename scalar_t>
    __device__
    bool ray_trace(vec3<scalar_t> s,
                   vec3<scalar_t> l,
                   const int bi,
                   const torch::PackedTensorAccessor64<scalar_t,4> d_mask,
                   const torch::PackedTensorAccessor64<scalar_t,4> d_hmap,
                   const torch::PackedTensorAccessor64<scalar_t,4> d_rechmap,
                   const torch::PackedTensorAccessor64<scalar_t,4> d_rgb,
                   vec3<scalar_t> &out) {
        bool ret = false;

        const int batch_size = d_rgb.size(0), h = d_rgb.size(2), w = d_rgb.size(3);
        scalar_t lx = l.x;
        scalar_t ly = l.y;
        scalar_t lh = l.z;

        scalar_t recx = s.x;
        scalar_t recy = s.y;
        scalar_t rech = s.z;

        scalar_t dirx = lx - recx, diry = ly - recy;
        bool gox = abs(dirx) > abs(diry);
        int searching_n = gox ? w : h;
        int starti = 0, endi = searching_n;
        if (lh > 0) {
            if (gox) {
                starti =  recx < lx ? recx:lx;

                endi = recx < lx ? lx:recx;
            }
            else {
                starti =  recy < ly ? recy:ly;
                endi = recy < ly ? ly:recy;
            }
        }
        if (lh < 0) {
            if (gox) {
                starti =  recx < lx ? 0:recx;
                endi = recx < lx ? recx:endi;
            }
            else {
                starti =  recy < ly ? 0:recy;
                endi = recy < ly ? recy:endi;
            }
        }

        scalar_t sx, sy;
        int flag = 0, last_flag = 0;
        for(int si = starti; si < endi; ++si) {
            /* Searching Point xyh */
            if (gox) {
                sx = si;
                sy = recy + (sx-recx)/dirx * diry;
            } else {
                sy = si;
                sx = recx + (sy-recy)/diry * dirx;
            }

            if (sx < 0 || sx > w-1 || sy < 0 || sy > h-1 || d_mask[bi][0][sy][sx] < 0.989) {
                last_flag = 0;
                continue;
            }


            scalar_t sh0 = d_hmap[bi][0][sy][sx];
            scalar_t sh1, sh;
            // do linear interpolation for sh; note that either sy or sx are floating number
            if (gox) {
                if ( sy+1 > h-1 || d_mask[bi][0][sy+1][sx] < 0.989)
                    sh = sh0;
                else {
                    sh1 = d_hmap[bi][0][sy+1][sx];
                    sh = lerp(sh0, sh1, sy - int(sy));
                }
            }
            else {
                if ( sx+1 > w-1 || d_mask[bi][0][sy][sx+1] < 0.989)
                    sh = sh0;
                else {
                    sh1 = d_hmap[bi][0][sy][sx+1];
                    sh = lerp(sh0, sh1, sx - int(sx));
                }
            }

            scalar_t dh = 1.0; // this controls the thickness; for double height map, dh = h_f - h_b
            bool intersect = check_intersect(recx, recy, rech, lx, ly, lh, sx, sy, sh, dh, flag);
            if (intersect) {
                /* TODO, which sampling? linear interpolation?  */
                out.x = d_rgb[bi][0][(int)sy][(int)sx];
                out.y = d_rgb[bi][1][(int)sy][(int)sx];
                out.z = d_rgb[bi][2][(int)sy][(int)sx];

                ret = true;
                break;
            }
            if (last_flag != 0){
                if (last_flag != flag) {
                    out.x = d_rgb[bi][0][(int)sy][(int)sx];
                    out.y = d_rgb[bi][1][(int)sy][(int)sx];
                    out.z = d_rgb[bi][2][(int)sy][(int)sx];

                    ret = true;
                    break;
                }
            }
            last_flag = flag;
        }

        return ret;
    }

    /*
    * Ray trace in the current scene
    * Given start point xyh, light point xyh, current receiver's height map,
    * Return:
    *      1. if intersect or not
    *      2. the color for the intersection point
    * */
    template <typename scalar_t>
    __device__
    bool ray_scene_intersect(vec3<scalar_t> ro,
                            vec3<scalar_t> rd,
                            const scalar_t dh,
                            const int bi,
                            const torch::PackedTensorAccessor64<scalar_t,4> d_mask,
                            const torch::PackedTensorAccessor64<scalar_t,4> d_hmap,
                            const torch::PackedTensorAccessor64<scalar_t,4> d_rgb,
                            vec3<scalar_t> &out) {
        bool ret = false;
        int h = d_mask.size(2);
        int w = d_mask.size(3);
        scalar_t dirx = rd.x, diry = rd.y, dirh = rd.z;

        /* Special Case, there's no direction update in x or y, but in h */
        if (abs(dirx) < 1e-6f && abs(diry) < 1e-6f) {
            out.x = d_rgb[bi][0][(int)ro.y][(int)ro.x];
            out.y = d_rgb[bi][1][(int)ro.y][(int)ro.x];
            out.z = d_rgb[bi][2][(int)ro.y][(int)ro.x];
            return true;
        }

        bool gox = abs(dirx) > abs(diry);
        int searching_n = gox ? w : h;

        scalar_t cur_h;
        scalar_t sx, sy;

        int prev_sign, cur_sign;

        // for(int si = starti; si < endi; ++si) {
        for(int si = 0; si < searching_n; ++si) {
            /* Searching Point XYH */
            if (gox) {
                sx = ro.x + si * sign(dirx);
                sy = ro.y + (sx-ro.x)/dirx * diry;
            } else {
                sy = ro.y + si * sign(diry);
                sx = ro.x + (sy-ro.y)/diry * dirx;
            }

            if (sx < 0 || sx > w-1 || sy < 0 || sy > h-1 || d_mask[bi][0][sy][sx] < 0.989) {
                continue;
            }

            scalar_t sh0 = d_hmap[bi][0][sy][sx];
            scalar_t sh1, sh;
            // do linear interpolation for sh; note that either sy or sx are floating number
            if (gox) {
                if (sy+1 > h-1 || d_mask[bi][0][sy+1][sx] < 0.989)
                    sh = sh0;
                else {
                    sh1 = d_hmap[bi][0][sy+1][sx];
                    sh = lerp(sh0, sh1, sy - int(sy));   // Always use 0.5 to do interpolation
                }

                cur_h = ro.z + (sx - ro.x) / dirx * dirh;
            }
            else {
                if ( sx + 1 > w-1 || d_mask[bi][0][sy][sx+1] < 0.989)
                    sh = sh0;
                else {
                    sh1 = d_hmap[bi][0][sy][sx+1];
                    sh = lerp(sh0, sh1, sx - int(sx));
                }

                cur_h = ro.z + (sy - ro.y) / diry * dirh;
            }

            // collide with the rechmap?
            if (si == 0) {  /* First sign */
                cur_sign = cur_h - sh; 
                continue;
            } else { 
                prev_sign = cur_sign;
            }

            cur_sign = cur_h - sh; 
            // if (cur_sign * prev_sign < 0.0 || abs(cur_sign) < dh) { /* pass through some objects */
            if (abs(cur_sign) < dh) { /* pass through some objects */
                out.x = d_rgb[bi][0][(int)sy][(int)sx];
                out.y = d_rgb[bi][1][(int)sy][(int)sx];
                out.z = d_rgb[bi][2][(int)sy][(int)sx];
                ret = true;
                break;
            }

        }

        return ret;
    }


    template <typename scalar_t>
    __global__ void hshadow_render_cuda_forward(
        const torch::PackedTensorAccessor64<scalar_t,4> d_rgb,
        const torch::PackedTensorAccessor64<scalar_t,4> d_mask,
        const torch::PackedTensorAccessor64<scalar_t,2> d_bb,
        const torch::PackedTensorAccessor64<scalar_t,4> d_hmap,
        const torch::PackedTensorAccessor64<scalar_t,4> d_rechmap,
        const torch::PackedTensorAccessor64<scalar_t,2> d_lightpos,
        torch::PackedTensorAccessor64<scalar_t,4> d_shadow) {
        const int wstride = gridDim.x * blockDim.x, hstride = gridDim.y * blockDim.y, bstride = gridDim.z * blockDim.z;
        const int batch_size = d_rgb.size(0), h = d_rgb.size(2), w = d_rgb.size(3);

        for (int bi = blockIdx.z; bi < batch_size; bi += bstride) {
            /* light xyh */
            scalar_t lx = d_lightpos[bi][0], ly = d_lightpos[bi][1], lh = d_lightpos[bi][2];
            int minh = max((int)d_bb[bi][0], 0), maxh = min((int)d_bb[bi][1], h-1), minw = max((int)d_bb[bi][2], 0), maxw = min((int)d_bb[bi][3], w-1);

            vec3<scalar_t> light(lx, ly, lh);
            for (int wi = blockIdx.x * blockDim.x + threadIdx.x;  wi < w; wi += wstride) for(int hi = blockIdx.y * blockDim.y + threadIdx.y;  hi < h; hi += hstride) {
                scalar_t shadow(1.0), mask_alpha(0.0);
                scalar_t recx = wi + 0.5, recy = hi+0.5, rech = d_rechmap[bi][0][hi][wi];

                vec3<scalar_t> start(recx, recy, rech);
                vec3<scalar_t> intersect_color;

                /* Searching Potentials */
                if (ray_trace(start, light, bi, d_mask, d_hmap, d_rechmap, d_rgb, intersect_color)) {
                    shadow = 0.0;
                }

                d_shadow[bi][0][hi][wi] = shadow;
                d_shadow[bi][1][hi][wi] = shadow;
                d_shadow[bi][2][hi][wi] = shadow;
            }
        }
    }


    template <typename scalar_t>
    __global__ void ray_intersect_cuda_forward(
        const torch::PackedTensorAccessor64<scalar_t,4> d_rgb,
        const torch::PackedTensorAccessor64<scalar_t,4> d_mask,
        const torch::PackedTensorAccessor64<scalar_t,4> d_hmap,
        const torch::PackedTensorAccessor64<scalar_t,4> d_rechmap,
        const torch::PackedTensorAccessor64<scalar_t,4> d_rd_map,
        torch::PackedTensorAccessor64<scalar_t,4> d_intersect) {

        const int wstride = gridDim.x * blockDim.x, hstride = gridDim.y * blockDim.y, bstride = gridDim.z * blockDim.z;
        const int batch_size = d_rgb.size(0), h = d_rgb.size(2), w = d_rgb.size(3);
        const scalar_t default_value = 0.0;

        for (int bi = blockIdx.z; bi < batch_size; bi += bstride) {
            for (int wi = blockIdx.x * blockDim.x + threadIdx.x;  wi < w; wi += wstride) for(int hi = blockIdx.y * blockDim.y + threadIdx.y;  hi < h; hi += hstride) {
                scalar_t shadow(1.0), mask_alpha(0.0);
                scalar_t recx = wi + 0.5, recy = hi+0.5, rech = d_rechmap[bi][0][hi][wi];

                scalar_t lx = d_rd_map[bi][0][hi][wi];
                scalar_t ly = d_rd_map[bi][1][hi][wi];
                scalar_t lh = d_rd_map[bi][2][hi][wi];

                vec3<scalar_t> start(recx, recy, rech);
                vec3<scalar_t> rd(lx, ly, lh);

                vec3<scalar_t> intersect_color;
                if (ray_trace(start, rd, bi, d_mask, d_hmap, d_rechmap, d_rgb, intersect_color)) {
                    d_intersect[bi][0][hi][wi] = intersect_color.x;
                    d_intersect[bi][1][hi][wi] = intersect_color.y;
                    d_intersect[bi][2][hi][wi] = intersect_color.z;
                    d_intersect[bi][3][hi][wi] = 1.0;
                } else {
                    d_intersect[bi][0][hi][wi] = default_value;
                    d_intersect[bi][1][hi][wi] = default_value;
                    d_intersect[bi][2][hi][wi] = default_value;
                    d_intersect[bi][3][hi][wi] = 0.0;
                }
            }
        }
    }

    template <typename scalar_t>
    __global__ void ray_scene_intersect_cuda_forward(
        const torch::PackedTensorAccessor64<scalar_t,4> d_rgb,
        const torch::PackedTensorAccessor64<scalar_t,4> d_mask,
        const torch::PackedTensorAccessor64<scalar_t,4> d_hmap,
        const torch::PackedTensorAccessor64<scalar_t,4> d_ro,
        const torch::PackedTensorAccessor64<scalar_t,4> d_rd,
        const scalar_t dh,
        torch::PackedTensorAccessor64<scalar_t,4> d_intersect) {

        const int wstride = gridDim.x * blockDim.x, hstride = gridDim.y * blockDim.y, bstride = gridDim.z * blockDim.z;
        const int batch_size = d_rgb.size(0), h = d_rgb.size(2), w = d_rgb.size(3);
        const scalar_t default_value = 0.0;

        for (int bi = blockIdx.z; bi < batch_size; bi += bstride) {
            for (int wi = blockIdx.x * blockDim.x + threadIdx.x;  wi < w; wi += wstride)  {
                for(int hi = blockIdx.y * blockDim.y + threadIdx.y;  hi < h; hi += hstride) {
                    // scalar_t rox = wi + 0.5, roy = hi+0.5, roh = d_ro[bi][0][hi][wi];
                    scalar_t rox = d_ro[bi][0][hi][wi];
                    scalar_t roy = d_ro[bi][1][hi][wi];
                    scalar_t roh = d_ro[bi][2][hi][wi];

                    scalar_t rdx = d_rd[bi][0][hi][wi];
                    scalar_t rdy = d_rd[bi][1][hi][wi];
                    scalar_t rdh = d_rd[bi][2][hi][wi];

                    vec3<scalar_t> ro(rox, roy, roh);
                    vec3<scalar_t> rd(rdx, rdy, rdh);

                    vec3<scalar_t> intersect_color;
                    if (ray_scene_intersect(ro, rd, dh, bi, d_mask, d_hmap, d_rgb, intersect_color)) {
                        d_intersect[bi][0][hi][wi] = intersect_color.x;
                        d_intersect[bi][1][hi][wi] = intersect_color.y;
                        d_intersect[bi][2][hi][wi] = intersect_color.z;
                        d_intersect[bi][3][hi][wi] = 1.0;
                    } else {
                        d_intersect[bi][0][hi][wi] = default_value;
                        d_intersect[bi][1][hi][wi] = default_value;
                        d_intersect[bi][2][hi][wi] = default_value;
                        d_intersect[bi][3][hi][wi] = 0.0;
                    }
                }
            }
        }
    }

    template <typename scalar_t>
    __global__ void reflect_render_cuda_forward(
        const torch::PackedTensorAccessor64<scalar_t,4> d_rgb,
        const torch::PackedTensorAccessor64<scalar_t,4> d_mask,
        const torch::PackedTensorAccessor64<scalar_t,4> d_hmap,
        const torch::PackedTensorAccessor64<scalar_t,4> d_rechmap,
        const torch::PackedTensorAccessor64<scalar_t,2> d_thresholds,
        torch::PackedTensorAccessor64<scalar_t,4> d_reflect,
        torch::PackedTensorAccessor64<scalar_t,4> d_reflect_height,
        torch::PackedTensorAccessor64<scalar_t,4> d_reflect_mask) {
        const int wstride = gridDim.x * blockDim.x, hstride = gridDim.y * blockDim.y, bstride = gridDim.z * blockDim.z;
        const int batch_size = d_rgb.size(0), h = d_rgb.size(2), w = d_rgb.size(3);

        for (int bi = blockIdx.z; bi < batch_size; bi += bstride) {
            for (int wi = blockIdx.x * blockDim.x + threadIdx.x;  wi < w; wi += wstride) for(int hi = blockIdx.y * blockDim.y + threadIdx.y;  hi < h; hi += hstride) {
                /*  Back tracing along the height
                    Find the closest point, filter the closest point
                */
                scalar_t min_dis = FLT_MAX;
                scalar_t min_r, min_g, min_b, min_height, min_mask;
                for(int ti = hi-1; ti >= 0; --ti) {
                    if (d_mask[bi][0][ti][wi] < 0.45)
                        continue;

                    scalar_t dis = abs(d_hmap[bi][0][ti][wi] * 2 + ti - hi);
                    if (dis < min_dis) {
                        min_dis = dis;
                        min_r = d_rgb[bi][0][ti][wi];
                        min_g = d_rgb[bi][1][ti][wi];
                        min_b = d_rgb[bi][2][ti][wi];

                        min_height = d_hmap[bi][0][ti][wi];
                        min_mask = d_mask[bi][0][ti][wi];
                    }
                }

                /* Check Condition */
                scalar_t cur_thresholds = d_thresholds[bi][0];
                if (min_dis < cur_thresholds) {
                    /* Let Use Nearest Neighbor First */
                    d_reflect[bi][0][hi][wi] = min_r;
                    d_reflect[bi][1][hi][wi] = min_g;
                    d_reflect[bi][2][hi][wi] = min_b;
                    d_reflect_height[bi][0][hi][wi] = min_height;
                    d_reflect_mask[bi][0][hi][wi] = 1.0;
                }

                // } else {
                //     scalar_t fadding = 1.0-(min_dis-cur_thresholds);
                //     if (fadding < 0.0) fadding = 0.0;
                //     d_reflect[bi][0][hi][wi] = min_r * fadding + (1.0-fadding);
                //     d_reflect[bi][1][hi][wi] = min_g * fadding + (1.0-fadding);
                //     d_reflect[bi][2][hi][wi] = min_b * fadding + (1.0-fadding);
                //     d_reflect_height[bi][0][hi][wi] = 0.0;
                //     d_reflect_mask[bi][0][hi][wi] = fadding;
                // }
            }
        }
    }

    template <typename scalar_t>
    __global__ void glossy_reflect_render_cuda_forward(
        const torch::PackedTensorAccessor64<scalar_t,4> d_rgb,
        const torch::PackedTensorAccessor64<scalar_t,4> d_mask,
        const torch::PackedTensorAccessor64<scalar_t,4> d_hmap,
        const torch::PackedTensorAccessor64<scalar_t,4> d_rechmap,
        const int sample_n,
        const float glossy,
        torch::PackedTensorAccessor64<scalar_t,4> d_reflect) {

        const int wstride = gridDim.x * blockDim.x, hstride = gridDim.y * blockDim.y, bstride = gridDim.z * blockDim.z;
        const int batch_size = d_rgb.size(0), h = d_rgb.size(2), w = d_rgb.size(3);

        for (int bi = blockIdx.z; bi < batch_size; bi += bstride) {
            for (int wi = blockIdx.x * blockDim.x + threadIdx.x;  wi < w; wi += wstride) for(int hi = blockIdx.y * blockDim.y + threadIdx.y;  hi < h; hi += hstride) {
                /*  Back tracing along the height
                    Find the closest point, filter the closest point
                */
                scalar_t min_dis = FLT_MAX;
                scalar_t min_r, min_g, min_b, min_height, min_mask;
                for(int ti = hi-1; ti >= 0; --ti) {
                    if (d_mask[bi][0][ti][wi] < 0.45)
                        continue;

                    scalar_t dis = abs(d_hmap[bi][0][ti][wi] * 2 + ti - hi);
                    if (dis < min_dis) {
                        min_dis = dis;
                        min_r = d_rgb[bi][0][ti][wi];
                        min_g = d_rgb[bi][1][ti][wi];
                        min_b = d_rgb[bi][2][ti][wi];

                        min_height = d_hmap[bi][0][ti][wi];
                        min_mask = d_mask[bi][0][ti][wi];
                    }
                }

                /* Check Condition */
                float cur_thresholds = 1e-1;
                if (min_dis < cur_thresholds) {
                    /* Let Use Nearest Neighbor First */
                    d_reflect[bi][0][hi][wi] = min_r;
                    d_reflect[bi][1][hi][wi] = min_g;
                    d_reflect[bi][2][hi][wi] = min_b;
                }
            }
        }
    }

} // namespace

std::vector<torch::Tensor> hshadow_render_cuda_forward(
    torch::Tensor rgb,
    torch::Tensor mask,
    torch::Tensor bb,
    torch::Tensor hmap,
    torch::Tensor rechmap,
    torch::Tensor light_pos) {
    const auto batch_size = rgb.size(0);
    const auto channel_size = rgb.size(1);
    const auto h = rgb.size(2);
    const auto w = rgb.size(3);
    const dim3 threads(16, 16, 1);
    const dim3 blocks((w + threads.x - 1) / threads.x, (h+threads.y-1)/threads.y, batch_size);
    torch::Tensor shadow_tensor = torch::ones({batch_size, 3, h, w}).to(rgb);

    AT_DISPATCH_FLOATING_TYPES(rgb.type(), "hshadow_render_cuda_forward", ([&] {
        hshadow_render_cuda_forward<scalar_t><<<blocks, threads>>>(
            rgb.packed_accessor64<scalar_t,4>(),
            mask.packed_accessor64<scalar_t,4>(),
            bb.packed_accessor64<scalar_t,2>(),
            hmap.packed_accessor64<scalar_t,4>(),
            rechmap.packed_accessor64<scalar_t,4>(),
            light_pos.packed_accessor64<scalar_t,2>(),
            shadow_tensor.packed_accessor64<scalar_t,4>());
    }));

    return {shadow_tensor};
}

std::vector<torch::Tensor> reflect_render_cuda_forward(
    torch::Tensor rgb,
    torch::Tensor mask,
    torch::Tensor hmap,
    torch::Tensor rechmap,
    torch::Tensor thresholds) {
    const auto batch_size = rgb.size(0);
    const auto channel_size = rgb.size(1);
    const auto h = rgb.size(2);
    const auto w = rgb.size(3);
    const dim3 threads(16, 16, 1);
    const dim3 blocks((w + threads.x - 1) / threads.x, (h+threads.y-1)/threads.y, batch_size);
    torch::Tensor reflection_tensor = torch::ones({batch_size, 3, h, w}).to(rgb);
    torch::Tensor reflection_mask_tensor = torch::zeros({batch_size, 1, h, w}).to(rgb);
    torch::Tensor reflection_height_tensor = torch::zeros({batch_size, 1, h, w}).to(rgb);

    AT_DISPATCH_FLOATING_TYPES(rgb.type(), "reflect_render_cuda_forward", ([&] {
        reflect_render_cuda_forward<scalar_t><<<blocks, threads>>>(
            rgb.packed_accessor64<scalar_t,4>(),
            mask.packed_accessor64<scalar_t,4>(),
            hmap.packed_accessor64<scalar_t,4>(),
            rechmap.packed_accessor64<scalar_t,4>(),
            thresholds.packed_accessor64<scalar_t,2>(),
            reflection_tensor.packed_accessor64<scalar_t,4>(),
            reflection_height_tensor.packed_accessor64<scalar_t,4>(),
            reflection_mask_tensor.packed_accessor64<scalar_t,4>());
    }));

    return {reflection_tensor, reflection_height_tensor,reflection_mask_tensor};
}


std::vector<torch::Tensor> glossy_reflect_render_cuda_forward(torch::Tensor rgb,
                                                              torch::Tensor mask,
                                                              torch::Tensor hmap,
                                                              torch::Tensor rechmap,
                                                              const int sample_n,
                                                              const float glossy) {
    const auto batch_size = rgb.size(0);
    const auto channel_size = rgb.size(1);
    const auto h = rgb.size(2);
    const auto w = rgb.size(3);
    const dim3 threads(16, 16, 1);
    const dim3 blocks((w + threads.x - 1) / threads.x, (h+threads.y-1)/threads.y, batch_size);

    torch::Tensor reflection_tensor = torch::ones({batch_size, 3, h, w}).to(rgb);

    AT_DISPATCH_FLOATING_TYPES(rgb.type(), "reflect_render_cuda_forward", ([&] {
        glossy_reflect_render_cuda_forward<scalar_t><<<blocks, threads>>>(
            rgb.packed_accessor64<scalar_t,4>(),
            mask.packed_accessor64<scalar_t,4>(),
            hmap.packed_accessor64<scalar_t,4>(),
            rechmap.packed_accessor64<scalar_t,4>(),
            sample_n,
            glossy,
            reflection_tensor.packed_accessor64<scalar_t,4>());
    }));

    return {reflection_tensor};

}


torch::Tensor ray_intersect_cuda_forward(torch::Tensor rgb,
                                        torch::Tensor mask,
                                        torch::Tensor hmap,
                                        torch::Tensor rechmap,
                                        torch::Tensor rd_map){
    const auto batch_size = rgb.size(0);
    const auto channel_size = rgb.size(1);
    const auto h = rgb.size(2);
    const auto w = rgb.size(3);
    const dim3 threads(16, 16, 1);
    const dim3 blocks((w + threads.x - 1) / threads.x, (h+threads.y-1)/threads.y, batch_size);

    torch::Tensor intersect_tensor = torch::ones({batch_size, 4, h, w}).to(rgb);

    AT_DISPATCH_FLOATING_TYPES(rgb.type(), "reflect_render_cuda_forward", ([&] {
        ray_intersect_cuda_forward<scalar_t><<<blocks, threads>>>(
            rgb.packed_accessor64<scalar_t,4>(),
            mask.packed_accessor64<scalar_t,4>(),
            hmap.packed_accessor64<scalar_t,4>(),
            rechmap.packed_accessor64<scalar_t,4>(),
            rd_map.packed_accessor64<scalar_t,4>(),
            intersect_tensor.packed_accessor64<scalar_t,4>());
    }));

    return intersect_tensor;

}


torch::Tensor ray_scene_intersect_cuda_forward(torch::Tensor rgb,
                                               torch::Tensor mask,
                                               torch::Tensor hmap,
                                               torch::Tensor ro,
                                               torch::Tensor rd,
                                               float dh){
    const auto batch_size = rgb.size(0);
    const auto channel_size = rgb.size(1);
    const auto h = rgb.size(2);
    const auto w = rgb.size(3);
    const dim3 threads(16, 16, 1);
    const dim3 blocks((w + threads.x - 1) / threads.x, (h+threads.y-1)/threads.y, batch_size);

    torch::Tensor intersect_tensor = torch::ones({batch_size, 4, h, w}).to(rgb);

    AT_DISPATCH_FLOATING_TYPES(rgb.type(), "reflect_render_cuda_forward", ([&] {
        ray_scene_intersect_cuda_forward<scalar_t><<<blocks, threads>>>(
            rgb.packed_accessor64<scalar_t,4>(),
            mask.packed_accessor64<scalar_t,4>(),
            hmap.packed_accessor64<scalar_t,4>(),
            ro.packed_accessor64<scalar_t,4>(),
            rd.packed_accessor64<scalar_t,4>(),
            dh,
            intersect_tensor.packed_accessor64<scalar_t,4>());
    }));

    return intersect_tensor;

}
