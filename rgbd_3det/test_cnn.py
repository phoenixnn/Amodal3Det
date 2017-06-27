"""
   Author: Zhuo Deng
   Date: Mar, 2016

   test a network on dataset
"""
import _init_paths
import caffe
import os.path as osp
import cPickle
from cnn.test import test_net


# load network
caffe.set_mode_gpu()
caffe.set_device(0)
prototxt = './models/test-19-bn.prototxt'
caffemodel = './output/rgbd_det_iter_40000.h5'
net = caffe.Net(prototxt, caffemodel, caffe.TEST)

# load pre-computed test data (.pkl file)
cache_file = osp.join('data', 'roidb_test_19.pkl')
if osp.exists(cache_file):
    with open(cache_file, 'rb') as fid:
        roidb = cPickle.load(fid)
    print 'data is loaded from {}'.format(cache_file)

print(len(roidb))

# test model
test_net(net, roidb)

print('done')