

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from cnn.config import cfg
import scipy.io as sio


def get_minibatch(roidb, num_classes):
    """
      Given selected images, construct a minibatch sampled from it.
    """
    num_images = len(roidb)

    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)

    # the number of rois per image
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images

    # the number of positive rois per image
    fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

    # Get the input images blob (n, channel, h, w), and scales formatted for caffe
    im_blob = _get_image_blob(roidb)

    # Now, build the region of interest and label blobs
    rois_blob = np.zeros((0, 5), dtype=np.float32)
    labels_blob = np.zeros((0), dtype=np.float32)

    if cfg.TRAIN.BBOX_REG_3d:
        bbox_3d_targets_blob = np.zeros((0, 7 * num_classes), dtype=np.float32)
        bbox_loss_3d_blob = np.zeros(bbox_3d_targets_blob.shape, dtype=np.float32)

    rois_context_blob = np.zeros((0, 5), dtype=np.float32)

    for im_i in xrange(num_images):
        labels, overlaps, im_rois, keep_inds, rois_context \
           = _sample_rois(roidb[im_i], fg_rois_per_image, rois_per_image)

        # Add to RoIs blob [xmin, ymin, xmax, ymax]
        # add batch id to rois as first column
        batch_ind = im_i * np.ones((im_rois.shape[0], 1))
        rois_blob_this_image = np.hstack((batch_ind, im_rois))

        # concatenate rois_blob
        rois_blob = np.vstack((rois_blob, rois_blob_this_image))

        # Add to labels, bbox targets, and bbox loss blobs
        labels_blob = np.hstack((labels_blob, labels))

        # context blob
        rois_context_blob_this_image = np.hstack((batch_ind, rois_context))
        rois_context_blob = np.vstack((rois_context_blob, rois_context_blob_this_image))

        if cfg.TRAIN.BBOX_REG_3d:
            targets_3d = roidb[im_i]['bbox_3d_targets']
            bbox_3d_targets, bbox_loss_3d_weights = \
                _get_bbox_3d_regression_labels(targets_3d[keep_inds, :], num_classes)
            bbox_3d_targets_blob = np.vstack((bbox_3d_targets_blob, bbox_3d_targets))
            bbox_loss_3d_blob = np.vstack((bbox_loss_3d_blob, bbox_loss_3d_weights))

    #
    blobs = {'img': im_blob,
             'rois': rois_blob,
             'labels': labels_blob}

    if cfg.TRAIN.BBOX_REG_3d:
        # get depth map blob
        dmap_blob = _get_dmap_blob(roidb)
        blobs['dmap'] = dmap_blob
        blobs['bbox_3d_targets'] = bbox_3d_targets_blob
        blobs['bbox_loss_3d_weights'] = bbox_loss_3d_blob

    blobs['rois_context'] = rois_context_blob

    return blobs


def _sample_rois(roidb, fg_rois_per_image, rois_per_image):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']
    rois_context = roidb['rois_context']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]

    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]
    rois_context = rois_context[keep_inds]

    return labels, overlaps, rois, keep_inds, rois_context


def _get_image_blob(roidb):
    """
    Builds an input blob from the images in the roidb
    """
    num_images = len(roidb)
    processed_ims = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        #sub-stract mean pixel value
        im = im.astype(np.float32, copy=False)
        im -= cfg.PIXEL_MEANS
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = np.zeros((num_images, processed_ims[0].shape[0], processed_ims[0].shape[1], 3),
                    dtype=np.float32)
    # fill the zero matrix above with actual image content respectively
    for i in xrange(num_images):
        im = processed_ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)

    return blob


def _get_dmap_blob(roidb):
    """
        build input depth map blob
    """
    num_images = len(roidb)
    processed_ims = []
    for i in xrange(num_images):
        tmp = sio.loadmat(roidb[i]['dmap'])
        dmap = tmp['dmap_f']
        if roidb[i]['flipped']:
            dmap = dmap[:, ::-1]
        # sub-stract mean pixel value
        dmap = dmap.astype(np.float32, copy=False)
        dmap -= cfg.PIXEL_MEANS_D
        processed_ims.append(dmap)

    # Create a blob to hold the input images
    blob = np.zeros((num_images, 3, processed_ims[0].shape[0], processed_ims[0].shape[1]),
                    dtype=np.float32)
    # fill the zero matrix above with actual image content respectively
    for i in xrange(num_images):
        im = processed_ims[i]
        blob[i, 0, :, :] = im
        blob[i, 1, :, :] = im
        blob[i, 2, :, :] = im

    return blob


# def _get_dmap_blob(roidb):
#     # hha
#     num_images = len(roidb)
#     processed_ims = []
#     for i in xrange(num_images):
#         im = cv2.imread(roidb[i]['dmap'])
#         if roidb[i]['flipped']:
#             im = im[:, ::-1, :]
#         #sub-stract mean pixel value
#         im = im.astype(np.float32, copy=False)
#         im -= cfg.PIXEL_MEANS_hha
#         processed_ims.append(im)
#
#     # Create a blob to hold the input images
#     blob = np.zeros((num_images, processed_ims[0].shape[0], processed_ims[0].shape[1], 3),
#                     dtype=np.float32)
#     # fill the zero matrix above with actual image content respectively
#     for i in xrange(num_images):
#         im = processed_ims[i]
#         blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
#     # Move channels (axis 3) to axis 1
#     # Axis order will become: (batch elem, channel, height, width)
#     channel_swap = (0, 3, 1, 2)
#     blob = blob.transpose(channel_swap)
#
#     return blob


def _get_bbox_3d_regression_labels(bbox_3d_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_loss_weights (ndarray): N x 4K blob of loss weights
    """
    clss = bbox_3d_target_data[:, 0]
    bbox_3d_targets = np.zeros((clss.size, 7 * num_classes), dtype=np.float32)
    bbox_loss_3d_weights = np.zeros(bbox_3d_targets.shape, dtype=np.float32)

    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 7 * cls
        end = start + 7
        bbox_3d_targets[ind, start:end] = bbox_3d_target_data[ind, 1:]
        bbox_loss_3d_weights[ind, start:end] = [1., 1., 1., 1., 1., 1., 1.]

    return bbox_3d_targets, bbox_loss_3d_weights