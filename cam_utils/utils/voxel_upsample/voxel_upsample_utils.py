import torch
from . import voxel_upsample_cuda

def scatter_point_inds(indices, point_inds, shape):
    ret = -1 * torch.ones(*shape, dtype=point_inds.dtype, device=point_inds.device)
    ndim = indices.shape[-1]
    flattened_indices = indices.view(-1, ndim).long()
    slices = [flattened_indices[:, i] for i in range(ndim)]
    ret[slices] = point_inds
    return ret

def generate_voxel2pinds(indices, spatial_shape):

    point_indices = torch.arange(indices.shape[0], device=indices.device, dtype=torch.int32)
    output_shape = list(spatial_shape)
    v2pinds_tensor = scatter_point_inds(indices, point_indices, output_shape)
    return v2pinds_tensor

def get_voxel_centers(voxel_coords, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float()
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers

def voxel_upsampler(voxel_scores, voxel_coords, pt_xyz, pt_coords, spatial_shape, voxel_size, 
                    point_cloud_range, nsample=16, radius=5, sample_range=3):

    v2p_inds = generate_voxel2pinds(voxel_coords, spatial_shape)
    voxel_xyz = get_voxel_centers(voxel_coords, voxel_size, point_cloud_range)

    """pt_coords = pt_coords.long()
    tmp = v2p_inds[pt_coords[:, 0], pt_coords[:, 1], pt_coords[:, 2]] 
    pt_scores = voxel_scores[tmp.long()]"""

    pt_scores = torch.zeros((pt_xyz.shape[0], nsample), dtype=torch.float32, device=voxel_scores.device)
    distance = torch.ones((pt_xyz.shape[0], nsample), dtype=torch.float32, device=voxel_scores.device)

    voxel_upsample_cuda.voxel_upsampler(pt_xyz.shape[0], spatial_shape[0], spatial_shape[1], spatial_shape[2], nsample, 
                                        radius, sample_range, sample_range, sample_range, pt_xyz, voxel_xyz,
                                        pt_coords, v2p_inds, voxel_scores, pt_scores, distance)

    neighbor_weights = torch.exp(- distance / 2)
    neighbor_weights = neighbor_weights / neighbor_weights.sum(dim=-1, keepdim=True)
    pt_scores = (pt_scores * neighbor_weights).sum(dim=-1)

    return pt_scores


def ssc(voxel_scores, voxel_coords, spatial_shape, voxel_size, 
        point_cloud_range, nsample=16, radius=5, sample_range=3):

    #v2p_inds = generate_voxel2pinds(voxel_coords, spatial_shape)
    voxel_xyz = get_voxel_centers(voxel_coords, voxel_size, point_cloud_range)

    """pt_coords = pt_coords.long()
    tmp = v2p_inds[pt_coords[:, 0], pt_coords[:, 1], pt_coords[:, 2]] 
    pt_scores = voxel_scores[tmp.long()]"""

    #pt_scores = torch.zeros((pt_xyz.shape[0], nsample), dtype=torch.float32, device=voxel_scores.device)
    #distance = torch.ones((pt_xyz.shape[0], nsample), dtype=torch.float32, device=voxel_scores.device)

    

    return voxel_xyz, voxel_scores