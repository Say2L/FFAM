#ifndef _STACK_VOXEL_QUERY_GPU_H
#define _STACK_VOXEL_QUERY_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int voxel_upsampler(int M, int R1, int R2, int R3, int nsample, float radius, 
    int z_range, int y_range, int x_range, at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, 
    at::Tensor new_coords_tensor, at::Tensor point_indices_tensor, at::Tensor voxel_score_tensor,
    at::Tensor pt_score_tensor, at::Tensor distance_tensor);


void voxel_upsampler_kernel_launcher(int M, int R1, int R2, int R3, int nsample,
    float radius, int z_range, int y_range, int x_range, const float *new_xyz, 
    const float *xyz, const int *new_coords, const int *point_indices, 
    const float *voxel_score, float *pt_score, float *distance);


#endif
