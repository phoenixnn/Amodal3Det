

import _init_paths
import caffe
import numpy as np
import cPickle
import os.path as osp
from cnn.train import train_net
from cnn.config import cfg
import argparse
import sys


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a RGBD R-CNN detection network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--setType', dest='setType',
                        help='image set type',
                        default='trainval', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    setType = args.setType
    output_dir = osp.join('output')

    # fix the random seeds (numpy and caffe) for reproducibility
    np.random.seed(cfg.RNG_SEED)
    caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe gpu
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(0)


    # load pre-computed raw training data (.pkl file) if exist
    cache_file = osp.join('data', 'roidb_' + setType + '_19.pkl')
    if osp.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            roidb = cPickle.load(fid)
        print 'raw data is loaded from {}'.format(cache_file)
    else:
        print "cache_file is {}".format(cache_file)

    print(len(roidb))

    # training a rgbd detection network
    print 'Output will be saved to `{:s}`'.format(output_dir)
    # solver_file = osp.join('models', 'solver.prototxt')
    train_net(args.solver, roidb, output_dir, pretrained_model=args.pretrained_model, max_iters=args.max_iters)