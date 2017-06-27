

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
# Consumers can get config by:
#   from m_config import cfg
cfg = __C

#
# Training options
#

__C.TRAIN = edict()

__C.TRAIN.max_iters = 40000

# Train bounding-box regressors 3d
__C.TRAIN.BBOX_REG_3d = True

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 2

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 256

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (427,)

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 10000

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5



#
# MISC
#

# detection classes
#__C.classes = ('background', 'bed', 'chair', 'sofa', 'table', 'toilet')
__C.classes = \
    ('background', 'bathtub',  'bed', 'bookshelf', 'box', 'chair', 'counter', 'desk', 'door', 'dresser',
     'garbage bin', 'lamp', 'monitor', 'night stand', 'pillow', 'sink', 'sofa', 'table', 'television', 'toilet')

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

__C.PIXEL_MEANS_D = 72.8123
# HHA
__C.PIXEL_MEANS_hha = np.array([[[126.4472, 94.2742, 132.5170]]])

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.DEDUP_BOXES = 1./16.
