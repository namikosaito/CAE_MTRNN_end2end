#!/usr/bin/env python
# coding:utf-8

import random, six, sys
import numpy as np
from chainer import cuda

def random_crop_image(img, size, test=False):
    #_, H, W = img.shape
    H, W, _ = img.shape

    ### crop input image with including target bbox
    min_top = 0
    min_left = 0
    max_top = H-size
    max_left = W-size

    if test:  
        dsize_x = img.shape[0] - size
        dsize_y = img.shape[1] - size
        top  = dsize_y // 2 #saito change 
        left = dsize_x // 2
        
    else:
        top  = random.randint(min_top, max_top)
        left = random.randint(min_left, max_left)

    bottom = top + size
    right  = left + size

    img = img[top:bottom, left:right, :]

    return img

