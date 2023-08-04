import os
import random

import numpy
import numpy as np

from skimage import io
import cv2
import torch
import imageio
import torch.utils.data as data
from skimage.restoration import denoise_tv_chambolle
from imageio import imwrite
from skimage import exposure
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import disk
from skimage.morphology import ball
from skimage.filters import rank, gaussian
from skimage import morphology
from skimage import segmentation


def equlization(im):
    eq_im = exposure.equalize_hist(im)
    # median = np.median(im)
    # im[im == im.min()] = median
    # im[im == im.max()] = median
    #
    # im = (im - im.min()) / (im.max() - im.min())
    # eq_im = rank.equalize(im, disk(20)) / 255
    return eq_im


class TEMDataset(data.Dataset):
    def __init__(self, data_path):
        super(TEMDataset, self).__init__()

        data_path = './Data/SAC_34-35 stack aligned.tif'
        im = io.imread(data_path)
        self.im = im

        # print("Total training examples:", len(self.im))

    def local_equlization(self, im):
        median = np.median(im)
        im[im == im.min()] = median
        im[im == im.max()] = median

        im = (im - im.min()) / (im.max() - im.min())
        eq_im = rank.equalize(im, disk(10)) / 255
        return eq_im

    def equlization(self, im):
        im = (im-im.min())/(im.max()-im.min())
        eq_im = exposure.equalize_hist(im)
        return eq_im
        # return (eq_im+im)/2

    def mask(self, im):
        mask1 = im <= 148500
        mask2 = im >= 158000
        # mask3 = im == 149622.62
        mask = mask1+mask2
        mask = morphology.binary_erosion(mask)
        mask = morphology.binary_erosion(mask)
        for i in range(35):
            mask = morphology.binary_dilation(mask)
        mask[:5,:]=1
        mask[-5:,:]=1
        return 1-mask


    def __getitem__(self, index):
        image = self.im[index]
        mask = self.mask(image)
        ori_image = self.equlization(image)
        image = self.local_equlization(image)
        image = np.expand_dims(image,0)

        return {'img': image, 'ori': ori_image, 'mask': mask, 'name': index}

    def __len__(self):
        return len(self.im)


