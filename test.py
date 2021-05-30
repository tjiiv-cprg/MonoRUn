import os
os.environ['PYTHONPATH'] = os.getcwd()
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '8'

import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Test (and evaluate)')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['bbox', 'bev', '3d'],
        help='evaluation metrics, e.g., "bbox", "bev", "3d"')
    parser.add_argument(
        '--result-dir', help='directory to save detection (and evaluation) results')
    parser.add_argument(
        '--val-set',
        action='store_true',
        help='whether to test validation set instead of test set')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--score-thr', type=float, default=0.3,
        help='bbox score threshold for visualization')
    parser.add_argument(
        '--extra', action='store_true',
        help='whether to draw extra results (covariance and reconstruction)')
    parser.add_argument(
        '--cov-scale', type=float, default=5.0,
        help='covariance scaling factor for visualization')
    args = parser.parse_args()
    return args


def args_to_str(args):
    argv = [args.config, args.checkpoint]
    if args.eval is not None:
        argv += ['--eval'] + args.eval
    if args.result_dir is not None:
        argv += ['--options',
                 'result_dir=' + args.result_dir,
                 'summary_file=' + os.path.join(args.result_dir, 'eval_results.txt')]
    if args.val_set:
        argv.append('--val-set')
    if args.show_dir is not None:
        argv += ['--show-dir', args.show_dir,
                 '--show-cov-scale', str(args.cov_scale)]
    if args.extra:
        argv.append('--show-extra')
    return argv


def main():
    # Todo: add visualization
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
    import tools.test
    sys.argv = [''] + args_to_str(args)
    tools.test.main()


if __name__ == '__main__':
    main()
