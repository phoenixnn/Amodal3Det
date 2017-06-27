from config import cfg
import numpy as np
import cv2
import scipy.io as sio
import utils.help_functions as mf
import heapq
from utils.timer import Timer

def _get_image_blob(im):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    blob = np.zeros((1, im.shape[0], im.shape[1], 3), dtype=np.float32)
    blob[0, 0:im.shape[0], 0:im.shape[1], :] = im_orig
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob


def _get_dmap_blob(dmap):
    blob = np.zeros((1, 3, dmap.shape[0], dmap.shape[1]), dtype=np.float32)
    dmap -= cfg.PIXEL_MEANS_D
    blob[0, 0, :, :] = dmap
    blob[0, 1, :, :] = dmap
    blob[0, 2, :, :] = dmap

    return blob

# def _get_dmap_blob(hha):
#     # hha
#     hha_orig = hha.astype(np.float32, copy=True)
#     hha_orig -= cfg.PIXEL_MEANS_hha
#     blob = np.zeros((1, hha.shape[0], hha.shape[1], 3), dtype=np.float32)
#     blob[0, 0:hha.shape[0], 0:hha.shape[1], :] = hha_orig
#     # Move channels (axis 3) to axis 1
#     # Axis order will become: (batch elem, channel, height, width)
#     channel_swap = (0, 3, 1, 2)
#     blob = blob.transpose(channel_swap)
#     return blob


def _get_rois_blob(bbox):
    """
       bbox: [xmin, ymin, xmax, ymax] N x 4

       return : [level, xmin, ymin, xmax, ymax] N x 5
    """
    levels = np.zeros((bbox.shape[0], 1), dtype=np.int)
    rois_blob = np.hstack((levels, bbox))

    return rois_blob.astype(np.float32, copy=False)


def _bbox_pred(boxes, box_deltas):
    """Transform the set of class-agnostic boxes into class-specific boxes
    by applying the predicted offsets (box_deltas)
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, box_deltas.shape[1]))

    boxes = boxes.astype(np.float, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + cfg.EPS
    heights = boxes[:, 3] - boxes[:, 1] + cfg.EPS
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = box_deltas[:, 0::4]
    dy = box_deltas[:, 1::4]
    dw = box_deltas[:, 2::4]
    dh = box_deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(box_deltas.shape)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def _bbox_pred_3d(bbox_3d, box_deltas_3d):
    """
    Args:
        bbox_3d: N x (n_cls * 7)
        box_deltas_3d: N x (n_cls * 7)

    Returns:
         pred_box_3d: N x (n_cls * 7)
    """
    if bbox_3d.shape[0] == 0:
        return np.zeros((0, box_deltas_3d.shape[1]))

    bbox_3d = bbox_3d.astype(np.float, copy=False)

    # N x n_cls
    cx = bbox_3d[:, 0::7]
    cy = bbox_3d[:, 1::7]
    cz = bbox_3d[:, 2::7]
    l = bbox_3d[:, 3::7]
    w = bbox_3d[:, 4::7]
    h = bbox_3d[:, 5::7]

    # offsets (N x n_cls)
    dx = box_deltas_3d[:, 0::7]
    dy = box_deltas_3d[:, 1::7]
    dz = box_deltas_3d[:, 2::7]
    dl = box_deltas_3d[:, 3::7]
    dw = box_deltas_3d[:, 4::7]
    dh = box_deltas_3d[:, 5::7]
    dt = box_deltas_3d[:, 6::7]

    # prediction
    pred_ctr_x = np.multiply(dx, l) + cx
    pred_ctr_y = np.multiply(dy, h) + cy
    pred_ctr_z = np.multiply(dz, w) + cz
    pred_l = np.multiply(np.exp(dl), l)
    pred_w = np.multiply(np.exp(dw), w)
    pred_h = np.multiply(np.exp(dh), h)
    pred_t = dt

    pred_boxes_3d = np.zeros(box_deltas_3d.shape)
    pred_boxes_3d[:, 0::7] = pred_ctr_x
    pred_boxes_3d[:, 1::7] = pred_ctr_y
    pred_boxes_3d[:, 2::7] = pred_ctr_z
    pred_boxes_3d[:, 3::7] = pred_l
    pred_boxes_3d[:, 4::7] = pred_w
    pred_boxes_3d[:, 5::7] = pred_h
    pred_boxes_3d[:, 6::7] = pred_t

    return pred_boxes_3d


def im_detect_3d(net, im, dmap, boxes, boxes_3d, rois_context):
    """  predict bbox and class scores
        bbox: N x 4 [xmin, ymin, xmax, ymax]
        bbox_3d : N x (n_cls * 7)
        return: scores: N x n_cls
                pred_boxes: N x (n_cls*4) [xmin, ymin, xmax, ymax]
                pred_boxes_3d: N x (n_cls*7)
    """
    # construct blobs
    blobs = {'img': None, 'dmap': None, 'rois': None}
    blobs['img'] = _get_image_blob(im)
    blobs['dmap'] = _get_dmap_blob(dmap)
    blobs['rois'] = _get_rois_blob(boxes)
    blobs['rois_context'] = _get_rois_blob(rois_context)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]
        boxes_3d = boxes_3d[index, :]
        blobs['rois_context'] = blobs['rois_context'][index, :]

    # reshape network inputs
    net.blobs['img'].reshape(*(blobs['img'].shape))
    net.blobs['dmap'].reshape(*(blobs['dmap'].shape))
    net.blobs['rois'].reshape(*(blobs['rois'].shape))
    net.blobs['rois_context'].reshape(*(blobs['rois_context'].shape))

    # forward pass for predictions
    blobs_out = net.forward(img=blobs['img'].astype(np.float32, copy=False),
                            dmap=blobs['dmap'].astype(np.float32, copy=False),
                            rois=blobs['rois'].astype(np.float32, copy=False),
                            rois_context=blobs['rois_context'].astype(np.float32, copy=False))

    # use softmax estimated probabilities
    scores = blobs_out['cls_prob']

    """ Apply bounding-box regression deltas """
    # 3d boxes
    box_deltas_3d = blobs_out['bbox_pred_3d']
    pred_boxes_3d = _bbox_pred_3d(boxes_3d, box_deltas_3d)

    #  2d boxes
    box_deltas = np.zeros((box_deltas_3d.shape[0], box_deltas_3d.shape[1]/7 *4))
    pred_boxes = _bbox_pred(boxes, box_deltas)

    if cfg.DEDUP_BOXES > 0:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]
        pred_boxes_3d = pred_boxes_3d[inv_index, :]

    return scores, pred_boxes, pred_boxes_3d


def test_net(net, roidb):
    """Test a network on an image database."""
    num_images = len(roidb)
    num_classes = len(cfg.classes)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
    max_per_set = 40 * num_images
    # heuristic: keep at most 100 detection per class per image prior to NMS
    max_per_image = 100
    # detection thresold for each class (this is adaptively set based on the
    # max_per_set constraint)
    thresh = -np.inf * np.ones(num_classes)
    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)
    top_scores = [[] for _ in xrange(num_classes)]
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    for i in xrange(num_images):
        # load image
        im = cv2.imread(roidb[i]['image'])

        # load dmap
        tmp = sio.loadmat(roidb[i]['dmap'])
        dmap = tmp['dmap_f']

        _t['im_detect'].tic()
        scores, boxes, boxes_3d = \
            im_detect_3d(net, im, dmap, roidb[i]['boxes'], roidb[i]['boxes_3d'], roidb[i]['rois_context'])
        _t['im_detect'].toc()

        _t['misc'].tic()
        # estimate threshold for each class
        for j in xrange(1, num_classes):
            inds = np.where((scores[:, j] > thresh[j]))[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_boxes_3d = boxes_3d[inds, j*7:(j+1)*7]

            top_inds = np.argsort(-cls_scores)[:max_per_image]
            # select top 'max_per_image'-th high scored boxes
            cls_scores = cls_scores[top_inds]
            cls_boxes = cls_boxes[top_inds, :]
            cls_boxes_3d = cls_boxes_3d[top_inds, :]

            # push new scores onto the minheap
            for val in cls_scores:
                heapq.heappush(top_scores[j], val)
            # if we've collected more than the max number of detection,
            # then pop items off the minheap and update the class threshold
            if len(top_scores[j]) > max_per_set:
                while len(top_scores[j]) > max_per_set:
                    heapq.heappop(top_scores[j])
                thresh[j] = top_scores[j][0]

            all_boxes[j][i] = \
                    np.hstack((cls_boxes, cls_scores[:, np.newaxis], cls_boxes_3d)).astype(np.float32, copy=False)

        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    for j in xrange(1, num_classes):
        print "thresh[j] = {}".format(thresh[j])
        for i in xrange(num_images):
            inds = np.where(all_boxes[j][i][:, 4] > thresh[j])[0]
            all_boxes[j][i] = all_boxes[j][i][inds, :]

    # save
    obj_arr = np.zeros((num_classes, num_images), dtype=np.object)
    for i in xrange(num_classes):
        for j in xrange(num_images):
            obj_arr[i][j] = all_boxes[i][j]

    sio.savemat('output/all_boxes_cells_test.mat', {'all_boxes': obj_arr})


