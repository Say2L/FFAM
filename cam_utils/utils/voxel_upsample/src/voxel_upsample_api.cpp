#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "voxel_upsample_gpu.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxel_upsampler", &voxel_upsampler, "voxel_upsampler");
}
