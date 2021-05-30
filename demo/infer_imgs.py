import os
os.environ['PYTHONPATH'] = os.getcwd()
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '8'

import argparse
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Infer from images in a directory')
    parser.add_argument('image_dir', help='directory of input images')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--calib', help='calibration matrix in .csv format',
                        default='demo/calib.csv')
    parser.add_argument(
        '--show-dir', 
        help='directory where painted images will be saved (default: $IMAGE_DIR/show)')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--extra', action='store_true',
        help='whether to draw extra results (covariance and reconstruction)')
    parser.add_argument(
        '--cov-scale', type=float, default=5.0, help='covariance scaling factor')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.gpu_ids is not None:
        gpu_ids = args.gpu_ids
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_ids = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_ids = [0]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpu_ids])
    if len(gpu_ids) != 1:
        raise NotImplementedError('multi-gpu testing is not yet supported')

    from mmcv.utils import track_iter_progress
    from monorun.apis import init_detector, inference_detector

    image_dir = args.image_dir
    assert os.path.isdir(image_dir)
    show_dir = args.show_dir
    if show_dir is None:
        show_dir = os.path.join(image_dir, 'show')
    if not os.path.exists(show_dir):
        os.makedirs(show_dir)
    calib = np.loadtxt(args.calib, delimiter=',').astype(np.float32)

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    if args.extra:
        model.test_cfg['rcnn']['debug'] = True

    img_list = os.listdir(image_dir)
    img_list.sort()
    for i, img_filename in enumerate(track_iter_progress(img_list)):
        img = cv2.imread(os.path.join(image_dir, img_filename))
        result = inference_detector(model, img, calib)
        model.show_result(
            img,
            calib,
            result,
            score_thr=args.score_thr,
            cov_scale=args.cov_scale,
            out_file=os.path.join(show_dir, img_filename))
    return


if __name__ == '__main__':
    main()
