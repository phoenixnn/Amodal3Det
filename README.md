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
3. [Usage](#usage)

## System requirements
The code is tested on the following system:
1. OS: Ubuntu 14.04
2. Hardware: Nvidia Titan X (GPU usage: ~9GB)
3. Software: Caffe, CUDA-7.5, cuDNN v4, Matlab 2015a, Anaconda2

## Basic Installation
1. clone the Amodal3Det repository: 
    ```Shell
    git clone https://github.com/phoenixnn/Amodal3Det.git

    ```
2. build Caffe:
    ```Shell
    # assume you clone the repo into the local your_root_dir
    cd your_root_dir
    make all -j8 && make pycaffe
    ```
3. install cuDNN:
    ```Shell
    sudo cp cudnn_folder/include/cudnn.h /usr/local/cuda-7.5/include/
    sudo cp cudnn_folder/lib64/*.so* /usr/local/cuda-7.5/lib64/
    ```

## Usage
1. Download NYUV2 dataset with 3D annotations:
    ```Shell
    wget 'url' -P your_root_dir/dataset/NYUV2/
    unzip your_root_dir/dataset/NYUV2/NYUV2_3D_dataset.zip
    ```
2. Download precomputed 2D segment proposals based on MCG3D:

Source Code and data are coming soon ...


