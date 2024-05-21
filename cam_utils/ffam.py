import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader
from spconv.pytorch.utils import PointToVoxel
from cam_utils.feature_factorization.nmf_utils import sparse_torch_dff
from cam_utils.utils.voxel_upsample.voxel_upsample_utils import voxel_upsampler
from cam_utils.cam_datasets import BaseDataset
from pcdet.models import build_network
from pcdet.models import load_data_to_gpu

def fix_batchnorm_parameters(model):
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm1d):
            layer.eval()


class FFAM:
    def __init__(self, model_config, data_config, model_ckpt_path, class_names, logger, **kwargs):
        super(FFAM, self).__init__()

        self.base_dataset = BaseDataset(data_config=data_config,
                                        class_names=class_names)
        self.max_num_points_per_voxel = data_config.DATA_PROCESSOR[2]['MAX_POINTS_PER_VOXEL']
        self.max_num_voxels = data_config.DATA_PROCESSOR[2]['MAX_NUMBER_OF_VOXELS']['test']
        self.voxel_size = data_config.DATA_PROCESSOR[2]['VOXEL_SIZE']
        self.point_cloud_range = data_config.POINT_CLOUD_RANGE

        self.model = build_network(model_cfg=model_config,
                                   num_class=len(class_names),
                                   dataset=self.base_dataset)
        
        self.model.load_params_from_file(filename=model_ckpt_path,
                                         logger=logger, to_cpu=True)
        self.model.cuda()
        self.model.eval()

        self.logger = logger

        self.model.backbone_3d.train()
        fix_batchnorm_parameters(self.model.backbone_3d)

    def target_loss(self, pred_boxes, pred_scores, obj_id=0):
        loss = torch.abs(pred_scores[obj_id]).sum() + torch.abs(pred_boxes[obj_id]).sum()
        return loss

    def norm_fn(self, x):
        norm_x = x - x.min()
        norm_x = norm_x / norm_x.max()
        return norm_x
    
    def compute_nmf(self, source_file_path, n_components=16):
        self.base_dataset.load_and_preprocess_pcl(source_file_path)

        dataloader = DataLoader(
            self.base_dataset, pin_memory=True, shuffle=False, 
            collate_fn=self.base_dataset.collate_batch, 
            drop_last=False, sampler=None, timeout=0
        )

        dataloader_iter = iter(dataloader)
        batch_dict = next(dataloader_iter)
        load_data_to_gpu(batch_dict)
        predictions = self.model(batch_dict)[0][0]

        points = batch_dict['points'][:, 1:].contiguous()
        sparse_output = batch_dict['multi_scale_3d_features']['x_conv3']
        self.feature_map_stride = batch_dict['multi_scale_3d_strides']['x_conv3']
        activations = sparse_output.features
        activations.retain_grad()
        concepts, nmf_map = sparse_torch_dff(activations, n_components=n_components)
        nmf_map = nmf_map.sum(dim=-1).detach()
        nmf_map = self.norm_fn(nmf_map)
        
        return nmf_map, predictions, points, activations, sparse_output
    
    def compute_batch_nmf(self, batch_dict, n_components=16):
        predictions = self.model(batch_dict)[0][0]

        points = batch_dict['points'][:, 1:].contiguous()
        sparse_output = batch_dict['multi_scale_3d_features']['x_conv3']
        self.feature_map_stride = batch_dict['multi_scale_3d_strides']['x_conv3']
        activations = sparse_output.features
        activations.retain_grad()
        concepts, nmf_map = sparse_torch_dff(activations, n_components=n_components)
        nmf_map = nmf_map.sum(dim=-1).detach()
        nmf_map = self.norm_fn(nmf_map)
        
        return nmf_map, predictions, points, activations, sparse_output
    
    def compute_ffam(self, nmf_map, points, activations, predictions, sparse_output, obj_id, retain_graph=True, sample_range=2, **kwargs):
        
        self.model.zero_grad()
        loss = self.target_loss(predictions['pred_boxes'], predictions['pred_scores'], obj_id)
        loss.backward(retain_graph=retain_graph)
        activation_gradients = deepcopy(activations.grad)
        activations.grad *= 0

        activation_gradients = torch.sum(activation_gradients.abs(), dim=-1)
        activation_gradients = self.norm_fn(activation_gradients)
        explanations =  activation_gradients * nmf_map

        voxel_size = np.array(self.voxel_size) * self.feature_map_stride
        self.voxel_generator = PointToVoxel(
            vsize_xyz=list(voxel_size),
            coors_range_xyz=list(self.point_cloud_range),
            num_point_features=3,
            max_num_points_per_voxel=self.max_num_points_per_voxel,
            max_num_voxels=self.max_num_voxels,
            device=points.device
        )
        
        _, vx_coord, _, pt_vx_id = self.voxel_generator.generate_voxel_with_id(points[:, :3].contiguous())
        valid_mask = pt_vx_id >= 0
        valid_points = deepcopy(points[valid_mask][:, :3]).contiguous()
        pt_coords = vx_coord[pt_vx_id[valid_mask]]

        sparse_indices = sparse_output.indices[:, 1:]
        cam_pt = voxel_upsampler(explanations, sparse_indices, valid_points, pt_coords, sparse_output.spatial_shape, 
                                 voxel_size, self.point_cloud_range, sample_range=sample_range)

        cam_pt = self.norm_fn(cam_pt)
        cam_res = cam_pt.new_zeros(points.shape[0])
        cam_res[valid_mask] = cam_pt

        return points.cpu().numpy(), cam_res.cpu().numpy()