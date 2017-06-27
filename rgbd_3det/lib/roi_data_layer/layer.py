"""
Author: Zhuo Deng
Date: Feb, 2016
m_RoIDataLayer implements a Caffe Python layer for input data.
"""

import caffe
from cnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import yaml


class m_RoIDataLayer(caffe.Layer):

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]

        return get_minibatch(minibatch_db, self._num_classes)

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        # para_str_ is defined in the train.prototxt, e.g., python_param {}
        layer_params = yaml.load(self.param_str_)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {
            'img': 0,
            'rois': 1,
            'labels': 2}

        # dataset blob: holds a batch of N images, each with 3 channels
        # The height and width (100 x 100) are dummy values
        top[0].reshape(1, 3, 100, 100)

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[1].reshape(1, 5)

        # labels blob: R categorical labels in [0, ..., K] for K foreground
        # classes plus background
        top[2].reshape(1)

        if cfg.TRAIN.BBOX_REG_3d:
            self._name_to_top_map['dmap'] = 3
            self._name_to_top_map['bbox_3d_targets'] = 4
            self._name_to_top_map['bbox_loss_3d_weights'] = 5
            # depth map blob:
            top[3].reshape(1, 3, 100, 100)

            # bbox_3d_targets blob: 7 targets per class
            top[4].reshape(1, self._num_classes * 7)

            # bbox_loss_3d_weights blob:
            top[5].reshape(1, self._num_classes * 7)

        self._name_to_top_map['rois_context'] = 6
        top[6].reshape(1, 5)


    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy dataset into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
