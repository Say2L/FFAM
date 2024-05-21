import argparse
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from cam_utils.ffam import FFAM
from cam_utils.vis_utils import visualize_attr_map

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model_cfg_file', type=str,
                        default='cfgs/kitti_models/second.yaml',
                        help='dataset/model config for the demo')
    parser.add_argument('--source_file_path', type=str, default='pcl_input/pcl_demo1.bin',
                        help='point cloud data file to analyze')
    parser.add_argument('--ckpt', type=str, default='pretrained/second_7862.pth',
                        help='path to pretrained model parameters')
    args = parser.parse_args()

    cfg_from_yaml_file(args.model_cfg_file, cfg)
  
    return args, cfg

def main(): 
    args, config = parse_config()
    logger = common_utils.create_logger()
    logger.info('------------------------ FFAM Demo -------------------------')

    cam = FFAM(model_config=config.MODEL,
               data_config=config.DATA_CONFIG,
               model_ckpt_path=args.ckpt, 
               class_names=config.CLASS_NAMES,
               logger=logger)
  
    logger.info('Start visual explanation computation:')

    nmf_map, predictions, raw_points, activations, sparse_output = cam.compute_nmf(args.source_file_path, n_components=64)

    base_det_boxes = predictions['pred_boxes'].cpu().detach().numpy()

    for obj_id in range(base_det_boxes.shape[0]):
        points, ffam = cam.compute_ffam(nmf_map, raw_points, activations,\
                                        predictions, sparse_output, \
                                        obj_id=obj_id, retain_graph=obj_id<(base_det_boxes.shape[0] - 1),
                                        sample_range=2)
        
        logger.info('Visualize explanation of {}-th object'.format(obj_id))
        
        visualize_attr_map(points, ffam, base_det_boxes)

if __name__ == '__main__':
    main()
