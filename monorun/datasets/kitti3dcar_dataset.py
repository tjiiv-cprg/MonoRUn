from .kitti3d_dataset import KITTI3DDataset

from mmdet.datasets import DATASETS


@DATASETS.register_module()
class KITTI3DCarDataset(KITTI3DDataset):
    CLASSES = ('Car', )
