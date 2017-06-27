"""
  Author: Zhuo Deng
  Date: Feb, 2016

  interface for training a rgbd detection network
"""

import caffe
import roi_data_layer.roidb as db_hammer
from utils.timer import Timer
import numpy as np
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2
from config import cfg

class SolverWrapper(object):

    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    # constructor
    def __init__(self, solver_prototxt, roidb, output_dir,
                 pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

        if cfg.TRAIN.BBOX_REG_3d:
            print("normalizing bbox 3d regression targets (bbox_3d_targets) ...")
            self.bbox_3d_means, self.bbox_3d_stds = db_hammer.normalize_bbox_3d_targets(roidb)

        # create a caffe solver instance
        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        # create a protocol buffer message
        self.solver_param = caffe_pb2.SolverParameter()

        # merge f.read() into self.solver_param
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        # set the input layer (the RoI dataset layer is designed by author)
        self.solver.net.layers[0].set_roidb(roidb)


    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        if cfg.TRAIN.BBOX_REG_3d:
            orig_0 = net.params['bbox_pred_3d'][0].data.copy()
            orig_1 = net.params['bbox_pred_3d'][1].data.copy()

            net.params['bbox_pred_3d'][0].data[...] = \
                    (net.params['bbox_pred_3d'][0].data * self.bbox_3d_stds[:, np.newaxis])
            net.params['bbox_pred_3d'][1].data[...] = \
                    (net.params['bbox_pred_3d'][1].data * self.bbox_3d_stds + self.bbox_3d_means)


        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # format filename
        filename = ('rgbd_3det_iter_{:d}'.format(self.solver.iter) + '.h5')
        filename = os.path.join(self.output_dir, filename)

        # save network
        #net.save(str(filename))
        net.save_hdf5(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG_3d:
            net.params['bbox_pred_3d'][0].data[...] = orig_0
            net.params['bbox_pred_3d'][1].data[...] = orig_1


    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()

            if self.solver.iter % (10 * self.solver_param.display) == 0:
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            # save model every "SNAPSHOT_ITERS" iterations
            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = self.solver.iter
                self.snapshot()

        # ensure the network is saved at last iteration
        if last_snapshot_iter != self.solver.iter:
            self.snapshot()


def train_net(solver_prototxt, roidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a rgbd detection network."""
    print('initializing network ...')
    sw = SolverWrapper(solver_prototxt, roidb, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    sw.train_model(max_iters)
    print 'done solving'