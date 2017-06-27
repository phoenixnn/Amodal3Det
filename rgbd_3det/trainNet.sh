#! /bin/bash

# save log file to 'rgbd_train.log'

python ./train_net_cmd.py --solver ./models/solver-19-bn.prototxt --setType trainval \
--weights ./models/rgbd_det_init_3d_allvgg.caffemodel 2>&1 | tee ./output/rgbd_train.log

