import os
import os.path as osp
import argparse
import numpy as np
import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate KITTI image metas')
    parser.add_argument('--data-root',
                        default='data/kitti',
                        help='root directory of KITTI object dataset')
    args = parser.parse_args()
    return args


def main():
    data_root = parse_args().data_root
    for data_type in ['training', 'testing']:
        print('Reading images from {} set...'.format(data_type))
        sub_dir = osp.join(data_root, data_type)
        img_dir = osp.join(sub_dir, 'image_2')
        meta_dir = osp.join(sub_dir, 'img_metas')
        if not osp.exists(meta_dir):
            os.mkdir(meta_dir)
        for filename in mmcv.track_iter_progress(os.listdir(img_dir)):
            img_idx = os.path.splitext(filename)[0]
            img = mmcv.imread(osp.join(img_dir, filename))
            shape = img.shape[:2]
            np.savetxt(osp.join(meta_dir, img_idx + '.txt'), shape, delimiter=',')
    print('Done!')
