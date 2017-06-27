

"""
pre-add necessary paths :
  1) caffe
  2) fast RCNN library

"""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

# get the directory where the called file stays
this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
add_path(osp.join(this_dir, '..', 'caffe-fast-rcnn', 'python'))

# Add lib to PYTHONPATH
add_path(osp.join(this_dir, 'lib'))


