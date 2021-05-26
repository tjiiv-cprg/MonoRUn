# MonoRUn

**MonoRUn: Monocular 3D Object Detection by Reconstruction and Uncertainty Propagation**. CVPR 2021. [[paper](https://arxiv.org/abs/2103.12605)]
Hansheng Chen, Yuyao Huang, Wei Tian*, Zhong Gao, Lu Xiong. (\*Corresponding author: Wei Tian.)

This repository is the PyTorch implementation for MonoRUn. The codes are based on [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMDetection3D](https://github.com/open-mmlab/mmdetection3d),  although we use our own data formats. The PnP C++ codes are modified from [PVNet](https://github.com/zju3dv/clean-pvnet).



<img src="demo/demo.gif" alt="demo"  />



## Installation

Please refer to [INSTALL.md](INSTALL.md).

## Data preparation

Download the official [KITTI 3D object dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d), including [left color images](http://www.cvlibs.net/download.php?file=data_object_image_2.zip), [calibration files](http://www.cvlibs.net/download.php?file=data_object_calib.zip) and [training labels](http://www.cvlibs.net/download.php?file=data_object_label_2.zip).

Download the train/val/test image lists [[Google Drive](https://drive.google.com/file/d/1edZKOKMV1Z8foip3QOMlnLte4DXuiJ0-/view?usp=sharing) | [Baidu Pan](https://pan.baidu.com/s/1nvk9ndmIdlruH8FEDL6bCA), password: `cj4u`]. For training with LiDAR supervision, download the preprocessed object coordinate maps [[Google Drive](https://drive.google.com/file/d/1yfgq0h0kzPQ6T6LEI3E0qQaIp_3mCVeZ/view?usp=sharing) | [Baidu Pan](https://pan.baidu.com/s/1kTJe-VA1Az1jhm3ctjVI5g), password: `fp3h`].

Extract the downloaded archives according to the following folder structure. It is recommended to symlink the dataset root to `$MonoRUn_ROOT/data`. If your folder structure is different, you may need to change the corresponding paths in config files.

```
$MonoRUn_ROOT
├── configs
├── monorun
├── tools
├── data
│   ├── kitti
│   │   ├── testing
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   └── test_list.txt
│   │   └── training
│   │       ├── calib
│   │       ├── image_2
│   │       ├── label_2
│   │       ├── obj_crd
│   │       ├── mono3dsplit_train_list.txt
│   │       ├── mono3dsplit_val_list.txt
│   │       └── trainval_list.txt
```

Run the preparation script to generate image metas:

```bash
cd $MonoRUn_ROOT
python tools/prepare_kitti.py
```

## Train

```bash
cd $MonoRUn_ROOT
```

To train without LiDAR supervision:

``` bash
python train.py configs/kitti_multiclass.py --gpu-ids 0 1
```

where  `--gpu-ids 0 1` specifies the GPU IDs. In the paper we use two GPUs for distributed training.  The number of GPUs affects the mini-batch size. You may change the `samples_per_gpu` option in the config file to vary the number of images per GPU.

To train with LiDAR supervision:

```bash
python train.py configs/kitti_multiclass_lidar_supv.py --gpu-ids 0 1
```

To view other training options:

```bash
python train.py -h
```

By default, logs and checkpoints will be saved to `$MonoRUn_ROOT/work_dirs`. You can run TensorBoard to plot the logs:

```bash
tensorboard --logdir $MonoRUn_ROOT/work_dirs
```

The default training data is the 3712-image training split. If you want to train on the full training set (train-val), please edit the config file and replace the line 

```python
ann_file=train_data_root + 'mono3dsplit_train_list.txt',
```

with

```python
ann_file=train_data_root + 'trainval_list.txt',
```

## Test

You can download the weights pretrained on the KITTI train split:

- `kitti_multiclass.pth` [[Google Drive](https://drive.google.com/file/d/1J_3BnMrhKGCeBT1R2VxK8hZbUhUlHpDd/view?usp=sharing) | [Baidu Pan](https://pan.baidu.com/s/1kYLP8YjYPi9QSQH2Y-XtrA), password: `6bih`]
- `kitti_multiclass_lidar_supv.pth` [[Google Drive](https://drive.google.com/file/d/1T0aTZtjs1YGU2j09VldLubUr_057p_eJ/view?usp=sharing) | [Baidu Pan](https://pan.baidu.com/s/1wzfQxnGH08RV9d0uCnZRwQ), password: `nmdb`]

To test and evaluate on the validation set using config at `$CONFIG_PATH` and checkpoint at `$CPT_PATH`:

```bash
python test.py $CONFIG_PATH $CPT_PATH --val-set --gpu-ids 0
```

To test on the test set and save detection results to `$RESULT_DIR`:

```bash
python test.py $CONFIG_PATH $CPT_PATH --result-dir $RESULT_DIR --gpu-ids 0
```

You can append the argument `--show-dir $SHOW_DIR` to save visualized results.

To view other testing options:

```bash
python test.py -h
```

Note: the training and testing scripts in the root directory are wrappers for the original scripts taken from MMDetection, which can be found in `$MonoRUn_ROOT/tools`. For advanced usage, please refer to the [official MMDetection docs](https://mmdetection.readthedocs.io).

## Demo

We provide a [demo script](demo/infer_imgs.py) to perform inference on images in a directory and save the visualized results. Example:

```bash
python demo/infer_imgs.py $KITTI_RAW_DIR/2011_09_30/2011_09_30_drive_0027_sync/image_02/data configs/kitti_multiclass_lidar_supv_trainval.py checkpoints/kitti_multiclass_lidar_supv_trainval.pth --calib demo/calib.csv --show-dir show/2011_09_30_drive_0027
```

## Citation

If you find this project useful in your research, please consider citing:

```
@inproceedings{monorun2021, 
  author = {Hansheng Chen and Yuyao Huang and Wei Tian and Zhong Gao and Lu Xiong}, 
  title = {MonoRUn: Monocular 3D Object Detection by Reconstruction and Uncertainty Propagation}, 
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  year = {2021}
}
```