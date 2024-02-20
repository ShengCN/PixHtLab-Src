
#include <torch/extension.h>

#include <vector>
#include <stdio.h>

// CUDA forward declarations
std::vector<torch::Tensor> plane_visualize_cuda(torch::Tensor planes, torch::Tensor camera, int h, int w);

// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> plane_visualize(torch::Tensor planes, torch::Tensor camera, int h, int w) {
    CHECK_INPUT(planes);
    CHECK_INPUT(camera);

    return plane_visualize_cuda(planes, camera, h, w);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &plane_visualize, "Plane Visualization (CUDA)");
}