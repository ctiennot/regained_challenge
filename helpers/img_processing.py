"""
This file is about processing the raw images:

    - cropping some parts of them on demand
    - resizing all images to the same size using center-crop, wrap or padding

"""

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.patches as patches


def load_files_names():
    pictures_train = {int(number[::-1][4:][::-1]): "data/pictures_train/" + number
                      for number in os.listdir("data/pictures_train")
                      if number[::-1][:4] == 'gpj.'}

    pictures_test = {int(number[::-1][4:][::-1]): "data/pictures_test/" + number
                      for number in os.listdir("data/pictures_test")
                      if number[::-1][:4] == 'gpj.'}

    return pictures_train, pictures_test


def global_view(img, method='center-crop', resize=None):
    """
    Return a global view (square size) of the image using the method given
    :param img: PIL.image object
    :param method: 'center-crop', 'warp' or 'Padding'
    :param resize: tuple, if provided resize the image to (w, h)
    :return: image
    """
    if method == 'center-crop':

        width, height = img.size   # Get dimensions
        new_width = new_height = min(img.size)
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2

        img = img.crop((left, top, right, bottom))

    if resize is not None:
        img = img.resize(resize)

    return img


def export_all_global(method='center-crop', resize=(500, 500), verbose=True):
    """Export all global views"""
    # save all training
    for i in range(1, 10001):
        if verbose and i%500==0:
            print "{}/{} views exported".format(i, 13000)
        global_view(Image.open(pictures_train[i]), method, resize).save(
            "data/global_views/{}.jpg".format(i))

    # save all testing
    for i in range(10001, 13001):
        if verbose and i%500==0:
            print "{}/{} views exported".format(i, 13000)
        global_view(Image.open(pictures_test[i]), method, resize).save(
            "data/global_views/{}.jpg".format(i))


def local_view(img, patch=(200, 200)):
    """ Return a randomly sampled patch from the image"""
    w_patch, h_patch = patch
    width, height = img.size
    w_rand = int(np.random.rand()*(width-w_patch))  # random width start
    h_rand = int(np.random.rand()*(height-h_patch))  # random height start
    return img.crop((w_rand, h_rand, w_rand+w_patch, h_rand+h_patch))


def export_all_local(method='center-crop', resize=(500, 500),
                     patch_per_img=10, verbose=True, seed=None):
    """
    Export all local views. Names are "id_k.jpg" where id is the id of the
    original image and k is the number of the patch (in 1:patch_per_img).
    :param patch_per_img: number of patch to create per image
    """
    if seed is not None:
        np.random.seed(seed)

    # save all training
    for i in range(1, 10001):
        if verbose and i%500==0:
            print "{}*{}/{} views exported".format(i, patch_per_img, 13000)
        img = Image.open(pictures_train[i])
        for k in range(patch_per_img):
            local_view(img, patch=(200, 200)).save(
                "data/local_views/{}_{}.jpg".format(i, k))

    # save all testing
    for i in range(10001, 13001):
        if verbose and i%500==0:
            print "{}*{}/{} views exported".format(i, patch_per_img, 13000)
        img = Image.open(pictures_test[i])
        for k in range(patch_per_img):
            local_view(img, patch=(200, 200)).save(
                "data/local_views/{}_{}.jpg".format(i, k))


if __name__ == '__ main__':

    # necessary global variables
    pictures_train, pictures_test = load_files_names()

    if False:
        export_all_global(resize=(100, 100))

    if False:
        export_all_local(patch_per_img=10, seed=1789)