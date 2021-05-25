import os
os.environ['PYTHONPATH'] = os.getcwd()
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '8'

import argparse
import cv2
import numpy as np
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description='Infer from webcam stream')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--device', help='device ID', default=0, type=int)
    parser.add_argument('--size', nargs=2,
                        metavar=('WIDTH', 'HEIGHT'), type=int,
                        help='size of the original frame',
                        default=(1280, 720))
    parser.add_argument('--crop', nargs=2,
                        metavar=('WIDTH', 'HEIGHT'), type=int,
                        help='size of the cropped frame',
                        default=(1280, 360))
    parser.add_argument('--fps', help='device FPS', type=float)
    parser.add_argument('--calib', help='calibration matrix in .csv format',
                        default='demo/calib.csv')
    parser.add_argument('--distort', help='distortion coefficients in .csv format',
                        default='demo/distort.csv')
    parser.add_argument('--fisheye', action='store_true',
                        help='whether to use OpenCV Fisheye model')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
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

    from monorun.apis import init_detector, inference_detector
    from monorun.core import draw_box_3d_pred, show_bev

    camera = cv2.VideoCapture(args.device)
    cmd = 'v4l2-ctl -d /dev/video{} -c sharpness=0'.format(args.device)
    proc = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    proc.wait()

    # load calibration
    calib = np.loadtxt(args.calib, delimiter=',').astype(np.float32)
    distort = np.loadtxt(args.distort, delimiter=', ').astype(np.float32)
    map_fun = cv2.fisheye.initUndistortRectifyMap if args.fisheye \
        else cv2.initUndistortRectifyMap
    width, height = args.size
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    crop_width, crop_height = args.crop
    assert crop_width <= width and crop_height <= height
    calib_crop = calib.copy()
    calib_crop[:2, 2] -= [(width - crop_width) / 2, (height - crop_height) / 2]
    undistort_map = map_fun(
        calib, distort, np.eye(3, 3),
        calib_crop, (crop_width, crop_height), cv2.CV_16SC2)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img_undistort = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
    if args.fps is not None:
        camera.set(cv2.CAP_PROP_FPS, args.fps)

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    model.test_cfg['rcnn']['debug'] = True

    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        camera.read(image=img)
        cv2.remap(img, undistort_map[0], undistort_map[1],
                  cv2.INTER_LINEAR, dst=img_undistort)
        result = inference_detector(model, img_undistort, calib_crop)

        img_pred_3d = img_undistort.copy()
        draw_box_3d_pred(
            img_pred_3d, result['bbox_3d_results'], calib_crop, score_thr=args.score_thr)
        viz_bev = show_bev(
            img_undistort, None, result['bbox_results'], result['bbox_3d_results'],
            result['oc_maps'], result['std_maps'], result['pose_covs'],
            calib_crop, scale=25, score_thr=args.score_thr,
            width=img_undistort.shape[1], height=img_undistort.shape[0],
            cov_scale=args.cov_scale)
        cv2.imshow('BEV', viz_bev)
        cv2.imshow('CameraView', img_pred_3d)
        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break
    return


if __name__ == '__main__':
    main()
