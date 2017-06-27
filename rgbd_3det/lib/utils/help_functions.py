"""
Author: Zhuo Deng
Date: Feb, 2016

Function Repository
"""

import numpy as np

def matstrcell2list(strcell):
    """
    Args:
        strcell: string cell .mat file

    Returns: a string list

    """
    words = [str(''.join(letter)) for letter_array in strcell for letter in letter_array]
    return words


def get_projection_mask():
    """

    Returns: a mask that defines the range in x and y axis
             Note: here index start from 0 and data[i:j]
                   doesn't include data[j]
    """
    #mask = np.ones((480, 640), dtype=np.int)
    #mask[45:471][41:601] = 0
    mask = (44, 471, 40, 601)
    return mask


def crop_image(img):
    """

    Args:
        img: image or n-D matrix

    Returns: cropped image by pre-defined mask

    """
    mask = get_projection_mask()
    im = img[mask[0]:mask[1], mask[2]:mask[3]]
    return im


def vis_depth_map(depth):
    """

    Args:
        depth: raw depth map

    Returns: uint8 map for visualization

    """
    max_dist = depth.max()
    gray_im = depth/max_dist * 255
    gray_im = gray_im.astype(np.uint8)
    return gray_im


def get_NYU_intrinsic_matrix():
    # standard cropped
    fx_rgb = 5.1885790117450188e+02
    fy_rgb = 5.1946961112127485e+02
    cx_rgb = 3.2558244941119034e+02
    cy_rgb = 2.5373616633400465e+02
    k = np.array([[fx_rgb, 0, cx_rgb - 40],
                  [0, fy_rgb, cy_rgb - 44],
                  [0, 0, 1]])

    return k