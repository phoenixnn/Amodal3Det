"""
    Author: Zhuo Deng
    Date: Feb, 2016
"""
import numpy as np
from cnn.config import cfg
import utils.cython_bbox
import math


def normalize_bbox_3d_targets(roidb):

    assert len(roidb) > 0
    num_images = len(roidb)
    num_classes = len(cfg.classes)

    # compute targets 3d
    for im_i in xrange(num_images):
        rois = roidb[im_i]['boxes']
        boxes_3d = roidb[im_i]['boxes_3d']
        max_overlaps = roidb[im_i]['max_overlaps']
        max_classes = roidb[im_i]['max_classes']
        roidb[im_i]['bbox_3d_targets'] = \
                _compute_targets_3d(rois, boxes_3d, max_overlaps, max_classes)

    # Compute values needed for means and stds
    # var(x) = E(x^2) - E(x)^2
    # note: 0 for background is ignored here
    class_counts = np.zeros((num_classes, 1)) + cfg.EPS
    sums = np.zeros((num_classes, 7))
    squared_sums = np.zeros((num_classes, 7))

    for im_i in xrange(num_images):
        targets = roidb[im_i]['bbox_3d_targets']
        # exclude the background class
        for cls in xrange(1, num_classes):
            cls_inds = np.where(targets[:, 0] == cls)[0]
            if cls_inds.size > 0:
                class_counts[cls] += cls_inds.size
                sums[cls, :] += targets[cls_inds, 1:].sum(axis=0)
                squared_sums[cls, :] += (targets[cls_inds, 1:] ** 2).sum(axis=0)

    means = sums / class_counts
    stds = np.sqrt(squared_sums / class_counts - means ** 2)

    # Normalize targets
    for im_i in xrange(num_images):
        targets = roidb[im_i]['bbox_3d_targets']
        for cls in xrange(1, num_classes):
            cls_inds = np.where(targets[:, 0] == cls)[0]
            roidb[im_i]['bbox_3d_targets'][cls_inds, 1:] -= means[cls, :]
            roidb[im_i]['bbox_3d_targets'][cls_inds, 1:] /= stds[cls, :]

    # These values will be needed for making predictions
    # (the predicts will need to be unnormalized and uncentered)
    return means.ravel(), stds.ravel()


def _compute_targets_3d(rois, boxes_3d, overlaps, labels):

    # Ensure ROIs are floats
    rois = rois.astype(np.float, copy=False)
    boxes_3d = boxes_3d.astype(np.float, copy=False)

    """ find positive 3d boxes (ex_boxes) and its gt (gt_boxes)
        based on 2d rois
    """
    # Indices of ground-truth ROIs
    gt_inds = np.where(overlaps == 1)[0]
    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = utils.cython_bbox.bbox_overlaps(rois[ex_inds, :],
                                                     rois[gt_inds, :])

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)

    gt_boxes = boxes_3d[gt_inds[gt_assignment], :]
    ex_boxes = boxes_3d[ex_inds, :]

    # normalized target (check later for h and w)
    targets_dx = (gt_boxes[:, 0] - ex_boxes[:, 0]) / (ex_boxes[:, 3] + cfg.EPS)
    targets_dy = (gt_boxes[:, 1] - ex_boxes[:, 1]) / (ex_boxes[:, 5] + cfg.EPS)
    targets_dz = (gt_boxes[:, 2] - ex_boxes[:, 2]) / (ex_boxes[:, 4] + cfg.EPS)
    targets_dl = np.log(gt_boxes[:, 3] / (ex_boxes[:, 3] + cfg.EPS))
    targets_dw = np.log(gt_boxes[:, 4] / (ex_boxes[:, 4] + cfg.EPS))
    targets_dh = np.log(gt_boxes[:, 5] / (ex_boxes[:, 5] + cfg.EPS))
    targets_dt = (gt_boxes[:, 6]) * math.pi / 180

    targets = np.zeros((boxes_3d.shape[0], 8), dtype=np.float32)
    targets[ex_inds, 0] = labels[ex_inds]
    targets[ex_inds, 1] = targets_dx
    targets[ex_inds, 2] = targets_dy
    targets[ex_inds, 3] = targets_dz
    targets[ex_inds, 4] = targets_dl
    targets[ex_inds, 5] = targets_dw
    targets[ex_inds, 6] = targets_dh
    targets[ex_inds, 7] = targets_dt

    return targets