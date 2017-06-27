
"""
  set up training set for 19 classes object detection
"""
import rgbd_3det._init_paths
import os.path as osp
import scipy.io as sio
from cnn.config import cfg
import PIL
import cPickle
import numpy as np
import math


def get_boxes_3d(gt_boxes_3d, prop_boxes, max_classes):
    """
    Args:
        gt_boxes_3d: N x 7
        prop_boxes: N x 140
        max_classes: class label start from 0

    Returns: boxes_3d  N x 7

    """
    n_boxes = len(max_classes)
    n_gt_boxes = gt_boxes_3d.shape[0]
    n_prop_boxes = prop_boxes.shape[0]
    assert(n_boxes == (n_gt_boxes + n_prop_boxes))

    cls = max_classes[n_gt_boxes:]
    props = np.zeros((0, 7), dtype=prop_boxes.dtype)
    for i in xrange(len(cls)):
        label = cls[i]
        sid = label*7
        eid = sid + 7
        tmp = prop_boxes[i, sid:eid]
        props = np.vstack((props, tmp))

    boxes_3d = np.vstack((gt_boxes_3d, props))

    return boxes_3d


def flip_boxes_3d(boxes_3d, K, width):

    # theta
    boxes_3d[:, -1] = -boxes_3d[:, -1]

    # cx
    cx = boxes_3d[:, 0]
    cz = boxes_3d[:, 2]
    ox = K[0, 2]
    fx = K[0, 0]
    x = cx*fx/(cz + cfg.EPS) + ox

    # flip x (x is start from 1)
    x1 = width - x + 1
    boxes_3d[:, 0] = (x1-ox)*cz/fx

    return

def get_context_rois(boxes):
    # center
    cx = (boxes[:, 0] + boxes[:, 2])/2.0
    cy = (boxes[:, 1] + boxes[:, 3])/2.0
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    # new box
    xmin = cx - 0.75*w
    xmin[np.where(xmin < 0)] = 0
    xmax = cx + 0.75*w
    xmax[np.where(xmax > 560)] = 560
    ymin = cy - 0.75*h
    ymin[np.where(ymin < 0)] = 0
    ymax = cy + 0.75*h
    ymax[np.where(ymax > 426)] = 426

    boxes_new = np.vstack((xmin, ymin, xmax, ymax))
    boxes_new = boxes_new.transpose()
    return boxes_new


if __name__ == '__main__':

    # pre-defined 19 classes plus background
    classes = tuple(cfg.classes)

    # load training image list
    nyu_data_path = osp.abspath('../../dataset/NYUV2')
    with open(osp.join(nyu_data_path, 'trainval.txt')) as f:
        imlist = f.read().splitlines()

    """  data construction """
    roidb = []
    # select the first kth proposals
    num_props = 2000
    # intrinsic matrix
    from utils.help_functions import get_NYU_intrinsic_matrix
    k = get_NYU_intrinsic_matrix()

    matlab_path = osp.abspath('../../matlab/NYUV2')
    for im_name in imlist:
        print(im_name)
        data = {}

        ''' --------------------ground truth----------------------------------- '''
        # gt boxes 3d: [x,y,z,l,w,h,theta]
        tmp = sio.loadmat(osp.join(matlab_path, 'gt_3D_19', str(int(im_name)) + '.mat'))
        gt_boxes_3d = tmp['gt_boxes_3d'].astype(np.float32)
        if gt_boxes_3d.shape[0] == 0:
            print 'no gt for target objects and skip.'
            continue

        # gt2Dsel [xmin, ymin, xmax, ymax]
        tmp = sio.loadmat(osp.join(matlab_path, 'gt_2Dsel_19', str(int(im_name)) + '.mat'))
        gt_boxes_sel = tmp['gt_boxes_sel'].astype(np.float32)
        num_gt_boxes = gt_boxes_sel.shape[0]

        # gt class ids
        tmp = sio.loadmat(osp.join(matlab_path, 'gt_label_19', str(int(im_name)) + '.mat'))
        gt_class_labels = tmp['gt_class_ids'].astype(np.float32)

        '''---------------------------proposals--------------------------------- '''
        # proposal 2d (N x 4)
        tmp = sio.loadmat(osp.join(matlab_path, 'proposal2d', str(int(im_name)) + '.mat'))
        boxes2d_prop = tmp['boxes2d_prop'].astype(np.float32)

        # proposal 3d (N x 140)
        tmp = sio.loadmat(osp.join(matlab_path, 'proposal3d', str(int(im_name)) + '.mat'))
        boxes3d_prop = tmp['boxes3d_prop'].astype(np.float32)

        '''---------------------------inputs----------------------------------- '''
        # image path
        data['image'] = osp.join(nyu_data_path, 'color', str(int(im_name)) + '.jpg')
        # depth map path (convert to [0, 255], 10m = 255)
        data['dmap'] = osp.join(matlab_path, 'dmap_f', str(int(im_name)) + '.mat')
        # HHA
        # data['dmap'] = osp.join(nyu_data_path, 'HHA', str(int(im_name)) + '.png')

        # rois 2d = gt2Dsel + proposal 2d
        data['boxes'] = np.vstack((gt_boxes_sel, boxes2d_prop[0:num_props-num_gt_boxes, :]))

        # overlap: compare rois proposals with gt_rois_selection
        tmp = sio.loadmat(osp.join(matlab_path, 'gt_overlaps_19', str(int(im_name)) + '.mat'))
        data['gt_overlaps'] = tmp['gt_overlaps'][0:num_props, :]
        gt_overlaps = data['gt_overlaps']

        # max_classes and max_overlaps
        max_overlaps = gt_overlaps.max(axis=1)
        max_classes = gt_overlaps.argmax(axis=1)
        data['max_classes'] = max_classes
        data['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

        # rois 3d (N x 7) = gt boxes 3d + proposal 3d
        data['boxes_3d'] = get_boxes_3d(gt_boxes_3d, boxes3d_prop[0:num_props-num_gt_boxes, :], max_classes)

        # flipped
        data['flipped'] = False

        # context
        boxes = data['boxes'].copy()
        data['rois_context'] = get_context_rois(boxes)

        roidb.append(data)


    """ Data augmentation """
    num_images = len(roidb)

    if cfg.TRAIN.USE_FLIPPED:
        widths = [PIL.Image.open(roidb[i]['image']).size[0]
                  for i in xrange(num_images)]
        print('flipping...')
        for i in xrange(num_images):
            print '{}image'.format(i)
            # flip 2d boxes
            boxes = roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2 - 1
            boxes[:, 2] = widths[i] - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()

            # flip 3d boxes
            boxes_3d = roidb[i]['boxes_3d'].copy()
            flip_boxes_3d(boxes_3d, k, widths[i])

            # flip box_context
            boxes_context = roidb[i]['rois_context'].copy()
            oldx1_11 = boxes_context[:, 0].copy()
            oldx2_22 = boxes_context[:, 2].copy()

            boxes_context[:, 0] = widths[i] - oldx2_22 - 1
            boxes_context[:, 2] = widths[i] - oldx1_11 - 1
            assert (boxes_context[:, 2] >= boxes_context[:, 0]).all()

            entry = {'image': roidb[i]['image'],
                     'boxes' : boxes,
                     'gt_overlaps' : roidb[i]['gt_overlaps'],
                     'flipped' : True,
                     'max_classes': roidb[i]['max_classes'],
                     'max_overlaps': roidb[i]['max_overlaps'],
                     'dmap': roidb[i]['dmap'],
                     'boxes_3d': boxes_3d,
                     'rois_context': boxes_context
                     }

            roidb.append(entry)

    print "total images: {}".format(len(roidb))

    print "all keys: {}".format(roidb[0].keys())
    # save training / test  data
    cache_file = 'roidb_trainval_19.pkl'
    with open(cache_file, 'wb') as fid:
        cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

    print "training data preparation is completed"