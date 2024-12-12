import os
import math
import pickle
import random
import numpy as np
import torch
import cv2
import imageio
from scipy.ndimage.interpolation import zoom

####################
# Files & IO
####################

###################### get image path list ######################
IMG_EXTENSIONS = ['tif', 'tiff']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    '''get image path list from image folder'''
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def _get_paths_from_lmdb(dataroot):
    '''get image path list from lmdb meta info'''
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), 'rb'))
    paths = meta_info['keys']
    sizes = meta_info['resolution']
    if len(sizes) == 1:
        sizes = sizes * len(paths)
    return paths, sizes


def get_image_paths(data_type, dataroot):
    '''get image path list
    support lmdb or image files'''
    paths, sizes = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            paths, sizes = _get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return paths, sizes


###################### read images ######################
def _read_img_lmdb(env, key, size):
    '''read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple'''
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img


def normalize_percentile(im, low=0, high=100):
    """Normalize the input 'im' by im = (im - p_low) / (p_high - p_low), where p_low/p_high is the 'low'th/'high'th percentile of the im
    Params:
        -im  : numpy.ndarray
        -low : float, typically 0.2
        -high: float, typically 99.8
    return:
        normalized ndarray of which most of the pixel values are in [0, 1]
    """

    p_low, p_high = np.percentile(im, low), np.percentile(im, high)
    return normalize_min_max(im, max_v=p_high, min_v=p_low)


def normalize_min_max(im, max_v, min_v=0):
    eps = 1e-10
    im = (im - min_v) / (max_v - min_v + eps)
    return im


def read_img(env, path, size=None):
    '''read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]'''
    if env is None:  # img
        img = imageio.volread(path)
        max = img.max()
    else:
        img = _read_img_lmdb(env, path, size)
    img = normalize_percentile(img, 0, 100)
    max2 = img.max()
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.ndim == 3:
        img = np.expand_dims(img, axis=-1)
    return img



####################
# image processing
# process on numpy image
####################


def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5
    size = len(img_list[0].shape)

    def _augment(img):
        if size == 5:
            if hflip:
                img = img[:, :, :, :, ::-1]
            if vflip:
                img = img[:, :, :, ::-1, :]
            if rot90:
                img = img.transpose(0, 1, 2, 4, 3)
            return img
        elif size == 4:
            if hflip:
                img = img[:, :, :, ::-1]
            if vflip:
                img = img[:, :, ::-1, :]
            if rot90:
                img = img.transpose(0, 1, 3, 2)
            return img

    return [_augment(img) for img in img_list]
