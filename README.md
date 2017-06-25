# Amodal Detection of 3D Objects: Inferring 3D Bounding Boxes from 2D Ones in RGB-Depth Images

By Zhuo Deng, Longin Jan Latecki (Temple University).
This paper was published in CVPR 2017.

## License 

Code is released under the GNU GENERAL PUBLIC LICENSE (refer to the LICENSE file for details).

## Cite The Paper
If you use this project for your research, please consider citing:

    @inproceedings{zhuo17amodal3det,
        author = {Zhuo Deng and Longin Jan Latecki},
        booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
        title = {Amodal Detection of 3D Objects: Inferring 3D Bounding Boxes from 2D Ones in RGB-Depth Images},
        year = {2017}
    }


## Contents
1. [System requirements](#system)
2. [Basic Installation](#install)

## System requirements
The code is tested on the following system:
1. OS: Ubuntu 14.04
2. Hardware: Nvidia Titan X (GPU usage: ~9GB)
3. Software: Caffe, CUDA-7.5, Cudnn v4, Matlab 2015a, Anaconda2

## Basic Installation
1. clone the Amodal3Det repository: 
    ```Shell
    git clone https://github.com/phoenixnn/Amodal3Det.git

    ```
2. build Caffe:
    ```Shell
    make all -j8 && make pycaffe
    ```


Source Code and data are coming soon ...


