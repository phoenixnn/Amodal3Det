import rgbd_3det._init_paths
import os.path as osp
import scipy.io as sio
import numpy as np
import cPickle

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

    # load training image list
    nyu_data_path = osp.abspath('../../dataset/NYUV2')
    with open(osp.join(nyu_data_path, 'test.txt')) as f:
        imlist = f.read().splitlines()

    """  data construction """
    roidb = []
    # select the first kth proposals
    num_props = 2000
    # intrinsic matrix
    from utils.help_functions import get_NYU_intrinsic_matrix
    k = get_NYU_intrinsic_matrix()

    #
    matlab_path = osp.abspath('../../matlab/NYUV2')

    for im_name in imlist:
        print(im_name)
        data = {}

        # image path
        data['image'] = osp.join(nyu_data_path, 'color', str(int(im_name)) + '.jpg')
        # depth map path (convert to [0, 255], 10m = 255)
        data['dmap'] = osp.join(matlab_path, 'dmap_f', str(int(im_name)) + '.mat')

        # proposal 2d (N x 4)
        tmp = sio.loadmat(osp.join(matlab_path, 'proposal2d', str(int(im_name)) + '.mat'))
        boxes2d_prop = tmp['boxes2d_prop'].astype(np.float32)

        # rois 2d =  proposal 2d
        data['boxes'] = boxes2d_prop[0:num_props, :]
        # proposal 3d (N x 140)
        tmp = sio.loadmat(osp.join(matlab_path, 'proposal3d', str(int(im_name)) + '.mat'))
        boxes3d_prop = tmp['boxes3d_prop'].astype(np.float32)
        data['boxes_3d'] = boxes3d_prop[0:num_props, :]

        # scene size
        boxes = data['boxes'].copy()
        data['rois_context'] = get_context_rois(boxes)

        roidb.append(data)

    print "total images: {}".format(len(roidb))

    # save training / test  data
    cache_file = 'roidb_test_19.pkl'
    with open(cache_file, 'wb') as fid:
        cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

    print "test data preparation is completed"