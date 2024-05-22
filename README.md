## ["FFAM: Feature Factorization Activation Map for Explanation of 3D Detectors"](https://arxiv.org/abs/2405.12601)
Thanks for the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), this implementation of the DCDet is mainly based on the pcdet v0.6.

Abstract: LiDAR-based 3D object detection has made impressive progress recently, yet most existing models are black-box, lacking interpretability. Previous explanation approaches primarily focus on analyzing image-based models and are not readily applicable to LiDAR-based 3D detectors. In this paper, we propose a Feature Factorization Activation Map (FFAM) to generate high-quality visual explanations for 3D detectors. FFAM employs non-negative matrix factorization to generate concept activation maps and subsequently aggregates these maps to obtain a global visual explanation. To achieve object-specific visual explanations, we refine the global visual explanation using the feature gradient of a target object. Additionally, we introduce a voxel upsampling strategy to align the scale between the activation map and input point cloud. We qualitatively and quantitatively analyze FFAM with multiple detectors on several datasets. Experimental results validate the high-quality visual explanations produced by FFAM.

### 1. Recommended Environment

- Linux (tested on Ubuntu 20.04)
- Python 3.6+
- PyTorch 1.1 or higher (tested on PyTorch 1.13)
- CUDA 9.0 or higher (tested on 11.6)

### 2. Set the Environment

```shell
pip install -r requirement.txt
python setup.py develop
cd cam_utils/utils/voxel_upsample & python setup.py develop

```

## Demo Example

Run the demo as follows:
```shell
python ffam_demo.py --model_cfg_file ${MODEL_CFG_PATH} \
    --source_file_path ${POINT_CLOUD_DATA} \
    --ckpt ${PRETRAINED_MODEL} \
```

## Acknowledgement
We thank the authors of [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet) for their open source release of their codebase.

## Paper

Please cite our paper if you find our work useful for your research:

```
@article{liu2024ffam,
  title={FFAM: Feature Factorization Activation Map for Explanation of 3D Detectors}, 
  author={Liu, Shuai and Li, Boyang and Fang, Zhiyu and Cui, Mingyue and Huang, Kai},
  journal={arXiv preprint arXiv:2405.12601},
  year={2024}
}
```
