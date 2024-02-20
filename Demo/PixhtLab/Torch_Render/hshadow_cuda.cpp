#include <torch/extension.h>

#include <vector>
#include <stdio.h>

// CUDA forward declarations
std::vector<torch::Tensor> hshadow_render_cuda_forward(torch::Tensor rgb, torch::Tensor mask, torch::Tensor mask_bb, torch::Tensor hmap, torch::Tensor rechmap, torch::Tensor light_pos);
std::vector<torch::Tensor> reflect_render_cuda_forward(torch::Tensor rgb, torch::Tensor mask, torch::Tensor hmap, torch::Tensor rechmap, torch::Tensor thresholds);
std::vector<torch::Tensor> glossy_reflect_render_cuda_forward(torch::Tensor rgb, torch::Tensor mask, torch::Tensor hmap, torch::Tensor rechmap, const int sample_n, const float glossy);
torch::Tensor ray_intersect_cuda_forward(torch::Tensor rgb, torch::Tensor mask, torch::Tensor hmap, torch::Tensor rechmap, torch::Tensor rd_map);
torch::Tensor ray_scene_intersect_cuda_forward(torch::Tensor rgb, torch::Tensor mask, torch::Tensor hmap, torch::Tensor ro, torch::Tensor rd, float dh);

// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/*  Heightmap Shadow Rendering 
    rgb:        B x 3 x H x W
    mask:       B x 1 x H x W
    mask:       B x 1 
    hmap:       B x 1 x H x W
    rechmap:    B x 1 x H x W
    light_pos:  B x 1 (x,y,h)
*/
std::vector<torch::Tensor> hshadow_render_forward(torch::Tensor rgb, torch::Tensor mask, torch::Tensor bb, torch::Tensor hmap, torch::Tensor rechmap, torch::Tensor light_pos) {
    CHECK_INPUT(rgb);
    CHECK_INPUT(mask);
    CHECK_INPUT(bb);
    CHECK_INPUT(hmap);
    CHECK_INPUT(rechmap);
    CHECK_INPUT(light_pos);

    return hshadow_render_cuda_forward(rgb, mask, bb, hmap, rechmap, light_pos);
}

std::vector<torch::Tensor> reflect_render_forward(torch::Tensor rgb, torch::Tensor mask, torch::Tensor hmap, torch::Tensor rechmap, torch::Tensor thresholds) {
    CHECK_INPUT(rgb);
    CHECK_INPUT(mask);
    CHECK_INPUT(hmap);
    CHECK_INPUT(rechmap);
    CHECK_INPUT(thresholds);

    return reflect_render_cuda_forward(rgb, mask, hmap, rechmap, thresholds);
}


std::vector<torch::Tensor> glossy_reflect_render_forward(torch::Tensor rgb,
                                                  torch::Tensor mask,
                                                  torch::Tensor hmap,
                                                  torch::Tensor rechmap,
                                                  int sample_n,
                                                  float glossy) {
    CHECK_INPUT(rgb);
    CHECK_INPUT(mask);
    CHECK_INPUT(hmap);
    CHECK_INPUT(rechmap);

    return glossy_reflect_render_cuda_forward(rgb, mask, hmap, rechmap, sample_n, glossy);
}


torch::Tensor ray_intersect_foward(torch::Tensor rgb,
                            torch::Tensor mask,
                            torch::Tensor hmap,
                            torch::Tensor rechmap,
                            torch::Tensor rd_map) {
    CHECK_INPUT(rgb);
    CHECK_INPUT(mask);
    CHECK_INPUT(hmap);
    CHECK_INPUT(rechmap);

    return ray_intersect_cuda_forward(rgb, mask, hmap, rechmap, rd_map);
}

torch::Tensor ray_scene_intersect_foward(torch::Tensor rgb,
                                         torch::Tensor mask,
                                         torch::Tensor hmap,
                                         torch::Tensor ro,
                                         torch::Tensor rd,
                                         float dh) {
    CHECK_INPUT(rgb);
    CHECK_INPUT(mask);
    CHECK_INPUT(hmap);
    CHECK_INPUT(ro);
    CHECK_INPUT(rd);

    return ray_scene_intersect_cuda_forward(rgb, mask, hmap, ro, rd, dh);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hshadow_render_forward, "Heightmap Shadow Rendering Forward (CUDA)");
    m.def("reflection", &reflect_render_forward, "Reflection Rendering Forward (CUDA)");
    m.def("glossy_reflection", &glossy_reflect_render_forward, "Glossy Reflection Rendering Forward (CUDA)");
    m.def("ray_intersect", &ray_intersect_foward, "Ray scene intersection");
    m.def("ray_scene_intersect", &ray_scene_intersect_foward, "Ray scene intersection");
}
