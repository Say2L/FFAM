import numpy as np
import pickle
import math
from pcdet.datasets import DatasetTemplate

class BaseDataset(DatasetTemplate):
    """
    OpenPCDet dataset to load and preprocess the point cloud
    """
    def __init__(self, data_config, class_names):
        """
        Parameters
        ----------
            data_config : EasyDict
               dataset cfg including data preprocessing properties (OpenPCDet)
            class_names :
                list of class names (OpenPCDet)
            ffam_config: EasyDict
                sampling properties for attribution map generation, see cfg file
        """
        super().__init__(dataset_cfg=data_config, class_names=class_names,
                         training=False)
        
    def load_and_preprocess_pcl(self, source_file_path):
        """
        load given point cloud file and preprocess data according OpenPCDet cfg

        Parameters
        ----------
        source_file_path : str
            path to point cloud to analyze (bin or npy)

        Returns
        -------
        pcl : ndarray (N, 4)
            preprocessed point cloud (x, y, z, intensity)
        """

        if source_file_path.split('.')[-1] == 'bin':
            points = np.fromfile(source_file_path, dtype=np.float32)
            points = points.reshape(-1, 4)
        elif source_file_path.split('.')[-1] == 'npy':
            points = np.load(source_file_path)
        else:
            raise NotImplementedError

        if self.dataset_cfg.FOV_POINTS_ONLY:
            angles = np.abs(np.degrees(np.arctan2(points[:, 1], points[:, 0])))
            mask = angles <= 42
            points = points[mask, :]

        input_dict = {
            'points': points
        }

        data_dict = self.prepare_data(data_dict=input_dict)
    
        self.points = data_dict['points']

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        
        input_dict = {
            'points': self.points,
        }

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict
    
    
class FFAMDataset(DatasetTemplate):
    """
    OpenPCDet dataset to load and preprocess the point cloud
    """
    def __init__(self, dataset_cfg, class_names, pseudo_gt_info_path, data_path, drop_order, drop_rate, dist_scale, logger):
        """
        Parameters
        ----------
            data_config : EasyDict
                dataset cfg including data preprocessing properties (OpenPCDet)
            class_names :
                list of class names (OpenPCDet)
            logger : Logger
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=False,
            root_path=None, logger=logger
        )
        self.data_path = data_path
        self.drop_order = drop_order
        self.drop_rate = drop_rate
        self.dist_scale = dist_scale
        self.logger = logger

        self.sample_files = []
        with open(pseudo_gt_info_path, 'rb') as f:
            self.pseudo_gt_infos = pickle.load(f)

    def __len__(self):
        return len(self.pseudo_gt_infos)

    @staticmethod
    def load_and_preprocess_pcl(sample_path):
        """
        load given point cloud file and preprocess data according OpenPCDet cfg

        Parameters
        ----------
        source_file_path : str
            path to point cloud to analyze (bin or npy)

        Returns
        -------
        pcl : ndarray (N, 4)
            preprocessed point cloud (x, y, z, intensity)
        """
        
        if sample_path.split('.')[-1] == 'bin':
            points = np.fromfile(sample_path, dtype=np.float32)
            points = points.reshape(-1, 4)
        elif sample_path.split('.')[-1] == 'npy':
            points = np.load(sample_path)
        else:
            raise NotImplementedError

        return points
    
    @staticmethod
    def get_points_in_box(points, gt_box):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        cx, cy, cz = gt_box[0], gt_box[1], gt_box[2]
        dx, dy, dz, rz = gt_box[3], gt_box[4], gt_box[5], gt_box[6]
        shift_x, shift_y, shift_z = x - cx, y - cy, z - cz
        
        MARGIN = 1e-1
        cosa, sina = math.cos(-rz), math.sin(-rz)
        local_x = shift_x * cosa + shift_y * (-sina)
        local_y = shift_x * sina + shift_y * cosa
        
        mask = np.logical_and(abs(shift_z) <= dz / 2.0, 
                            np.logical_and(abs(local_x) <= dx / 2.0 + MARGIN, 
                                            abs(local_y) <= dy / 2.0 + MARGIN))
        
        return mask

    def __getitem__(self, index):
        sample_path = self.data_path + '/' + self.pseudo_gt_infos[index]['frame_id'] + '.npy'
        raw_points = self.load_and_preprocess_pcl(sample_path)
        raw_ffam = self.pseudo_gt_infos[index]['ffam']
      
        pseudo_gt_box = np.array(self.pseudo_gt_infos[index]['pseudo_gt_box'])
        diag_len = np.sqrt(np.sum((pseudo_gt_box[3:5])**2))
        center_dist = np.sqrt(np.sum((raw_points[:, :2] - pseudo_gt_box[None, :2])**2, axis=-1))
        dist_mask = center_dist < self.dist_scale * diag_len
        #dist_mask = self.get_points_in_box(raw_points, pseudo_gt_box)
        points = raw_points[dist_mask]
        ffam = raw_ffam[dist_mask]

        point_indices = np.argsort(ffam)
        if self.drop_order == 'low_drop':
            start_ind = int(points.shape[0] * self.drop_rate)
            point_indices = point_indices[start_ind:]
            
        elif self.drop_order == 'high_drop':
            end_ind = int(points.shape[0] * (1 - self.drop_rate))
            point_indices = point_indices[:end_ind]
            
        elif self.drop_order == 'random':
            indices = np.random.permutation(len(points))
            end_ind = int(points.shape[0] * (1 - self.drop_rate))
            point_indices = indices[:end_ind]

        input_dict = {
            'points': np.concatenate((points[point_indices], raw_points[~dist_mask]), axis=0),
            'attr_map': np.concatenate((ffam[point_indices], raw_ffam[~dist_mask]), axis=0),
            'frame_id': self.pseudo_gt_infos[index]['frame_id'],
            'pseudo_gt_box': np.array(self.pseudo_gt_infos[index]['pseudo_gt_box']),
            'pseudo_gt_label': np.array(self.pseudo_gt_infos[index]['pseudo_gt_label']),
            'pseudo_gt_score': np.array(self.pseudo_gt_infos[index]['pseudo_gt_score']),
        }

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict