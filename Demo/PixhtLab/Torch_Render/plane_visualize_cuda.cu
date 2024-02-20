#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
	template <typename scalar_t>
	struct vec3 {
		scalar_t x, y, z;

		__device__ __host__
		vec3<scalar_t>():x(0),y(0),z(0) {}

		__device__ __host__
		vec3<scalar_t>(scalar_t a):x(a),y(a),z(a) {}

		__device__ __host__
		vec3<scalar_t>(scalar_t xx, scalar_t yy, scalar_t zz):x(xx),y(yy),z(zz) {}

		__device__ __host__
		vec3<scalar_t> operator*(const scalar_t &rhs) const {
			vec3<scalar_t> ret(x,y,z);
			ret.x = x * rhs;
			ret.y = y * rhs;
			ret.z = z * rhs;
			return ret;
		}

		__device__ __host__
		vec3<scalar_t> operator/(const scalar_t &rhs) const {
			vec3<scalar_t> ret(x,y,z);
			ret.x = x / rhs;
			ret.y = y / rhs;
			ret.z = z / rhs;
			return ret;
		}

		__device__ __host__
		vec3<scalar_t> operator+(const vec3<scalar_t> &rhs) const {
			vec3<scalar_t> ret(x,y,z);
			ret.x = x + rhs.x;
			ret.y = y + rhs.y;
			ret.z = z + rhs.z;
			return ret;
		}

		__device__ __host__
		vec3<scalar_t> operator-(const vec3<scalar_t> &rhs) const {
			vec3<scalar_t> ret(x,y,z);
			ret.x = x - rhs.x;
			ret.y = y - rhs.y;
			ret.z = z - rhs.z;
			return ret;
		}
	};

	template <typename scalar_t>
	struct Ray {
		vec3<scalar_t> ro, rd;
	};

	template <typename scalar_t>
	struct Scene {
		vec3<scalar_t> pp, pn;
	};

	template <typename scalar_t>
	__device__
	float deg2rad(scalar_t d) {
		return d/180.0 * 3.1415926f;
	}

	template <typename scalar_t>
	__device__ 
	scalar_t dot(vec3<scalar_t> a, vec3<scalar_t> b) {
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	template <typename scalar_t>
	__device__ 
	vec3<scalar_t> cross(vec3<scalar_t> a, vec3<scalar_t> b) {
		vec3<scalar_t> ret(0.0f);
		ret.x = a.y * b.z - a.z * b.y;
		ret.y = a.z * b.x - a.x * b.z;
		ret.z = a.x * b.y - a.y * b.x;
		return ret;
	}

	template <typename scalar_t>
	__device__ 
	scalar_t length(vec3<scalar_t> a) {
		return sqrt(dot(a, a));
	}

	template <typename scalar_t>
	__device__ 
	vec3<scalar_t> normalize(vec3<scalar_t> a) {
		return a/length(a);
	}

	template <typename scalar_t>
	__device__ 
	scalar_t get_focal(int w, scalar_t fov) {
		return 0.5 * w / tan(deg2rad(fov * 0.5));
	}

	template <typename scalar_t>
	__device__ 
	vec3<scalar_t> get_rd(vec3<scalar_t> right, vec3<scalar_t> front, vec3<scalar_t> up, int h, int w, scalar_t focal, scalar_t x, scalar_t y) {
		/* x, y in [-1, 1] */
		right = normalize(right);
		front = normalize(front); 
		up = normalize(up);

		return front * focal + right * x * (float)w * 0.5f + up * y * (float)h * 0.5f;
	}

	template <typename scalar_t>
	__device__ 
	Ray<scalar_t> get_ray(int h, int w, float hi, float wi, vec3<scalar_t> right, vec3<scalar_t>front, vec3<scalar_t> up, vec3<scalar_t> cam_pos, float fov) {
		/* Note, wi/hi is in [-1.0, 1.0] */
		Ray<scalar_t> ray;

		float focal = 0.5f * w / tan(deg2rad(fov)); 
		ray.ro = cam_pos;
		ray.rd = front * focal + right * 0.5f * w * wi + up * 0.5f * h * hi;
		return ray;
	}
	
	template <typename scalar_t>
	__device__
	bool plane_intersect(Ray<scalar_t> ray, vec3<scalar_t> p, vec3<scalar_t> n, float &t) {
		vec3<scalar_t> ro = ray.ro, rd = ray.rd;
		t = dot(p-ro, n)/dot(rd, n);
		return t >= 0.0;
	}

	template <typename scalar_t>
	__device__
	vec3<scalar_t> horizon2front(scalar_t horizon, int h, int w, float fov) {
		scalar_t yoffset =  h / 2 - horizon;
		scalar_t focal = 0.5f * w / tan(deg2rad(fov));
		vec3<scalar_t> front = vec3<scalar_t>(0.0f,0.0f,-1.0f) * focal + vec3<scalar_t>(0.0f, 1.0f, 0.0f) * yoffset;
		return normalize(front); 
	}

	template <typename scalar_t>
	__device__
	vec3<scalar_t> plane_texture(vec3<scalar_t> p) {
		float freq = 6.0f;
		float u = sin(p.x * freq), v = sin(p.z * freq);
		vec3<scalar_t> ret(0.0f);
		float line_width = 0.05f; 
		if ((abs(u) < line_width || abs(v) < line_width)) {
			ret = vec3<scalar_t>(1.0f);
		}
		return ret;
	}

	template <typename scalar_t>
	__device__
	bool ray_scene_trace(Ray<scalar_t> ray, Scene<scalar_t> scene, vec3<scalar_t> &color) {
		color = vec3<scalar_t>(0.0f);
		float t;
		if (plane_intersect(ray, scene.pp, scene.pn, t)) {
			vec3<scalar_t> intersect_pos = ray.ro + ray.rd * t;
			color = plane_texture(intersect_pos);
			return true;
		}
		return false;
	}

	template <typename scalar_t>
	__global__ void plane_visualize_foward(
		const torch::PackedTensorAccessor64<scalar_t,2> d_plane,
		const torch::PackedTensorAccessor64<scalar_t,2> d_camera,
        torch::PackedTensorAccessor64<scalar_t,4> d_vis) {
		const int wstride = gridDim.x * blockDim.x, hstride = gridDim.y * blockDim.y, bstride = gridDim.z * blockDim.z;
		const int batch_size = d_vis.size(0), h = d_vis.size(2), w = d_vis.size(3);
		const int samples = 10;

		vec3<scalar_t> cam_pos(0.0f, 1.0f, 1.0f), front(0.0f,0.0f,-1.0f), right(1.0f, 0.0f, 0.0f), up(0.0f, 1.0f, 0.0f);
		for (int bi = blockIdx.z; bi < batch_size; bi += bstride) {
			// scalar_t px = d_plane[bi][0], py = d_plane[bi][1], pz = d_plane[bi][2];
			vec3<scalar_t> plane_pos(d_plane[bi][0], d_plane[bi][1], d_plane[bi][2]);
			vec3<scalar_t> plane_norm(d_plane[bi][3], d_plane[bi][4], d_plane[bi][5]);
			Scene<scalar_t> scene = {plane_pos, plane_norm};

			scalar_t fov = d_camera[bi][0], horizon = d_camera[bi][1]; 
			front = normalize(horizon2front(horizon, h, w, fov));
			up = normalize(cross(right, front));
			for (int wi = blockIdx.x * blockDim.x + threadIdx.x; wi < w; wi += wstride) 
                for(int hi = blockIdx.y * blockDim.y + threadIdx.y; hi < h; hi += hstride) {
					bool intersect = false;
					vec3<scalar_t> color(0.0f);
					for (int si = 0; si < samples * samples; ++si) {
						float hoffset = (float)(si/samples)/max(samples-1, 1);
						float woffset = (float)(si%samples)/max(samples-1, 1);
						float x = (float)(wi + woffset)/w * 2.0 - 1.0;
						float y = (float)(hi + hoffset)/h * 2.0 - 1.0;
						Ray<scalar_t> ray = get_ray(h, w, y, x, right, front, up, cam_pos,fov);
						vec3<scalar_t> tmp_color(0.0f);
						if(ray_scene_trace(ray, scene, tmp_color)) {
							color = color + tmp_color;
							intersect = intersect || true;
						}
					}
					if (intersect) {
						color = color / float(samples * samples);
						d_vis[bi][0][hi][wi] = color.x;
						d_vis[bi][1][hi][wi] = color.y;
						d_vis[bi][2][hi][wi] = color.z;
					}
            }
        }
    }

} // namespace

std::vector<torch::Tensor> plane_visualize_cuda(torch::Tensor planes, torch::Tensor camera, int h, int w){
	const auto batch_size = planes.size(0);
	const int threads = 512;
	const dim3 blocks((w + threads - 1) / threads, (h+threads-1)/threads, batch_size);

	torch::Tensor vis_tensor = torch::zeros({batch_size, 3, h, w}).to(planes);
	AT_DISPATCH_FLOATING_TYPES(planes.type(), "plane_visualize_foward", ([&] {
		plane_visualize_foward<scalar_t><<<blocks, threads>>>(
			planes.packed_accessor64<scalar_t,2>(),
			camera.packed_accessor64<scalar_t,2>(),
			vis_tensor.packed_accessor64<scalar_t,4>()
		);
	}));

	return {vis_tensor};
}
