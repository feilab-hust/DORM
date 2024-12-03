import os
import sys
import time
import math
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import torch
from torchvision.utils import make_grid
from shutil import get_terminal_size
import tifffile
import skimage

import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

def save_img(img, mi, ma, img_path, scale=1, mode='8bit'):
    img = np.expand_dims(img, 1)
    img = reverse_norm(img, mi, ma, scale, mode)
    tifffile.imwrite(img_path, img, imagej=True)
    mip_xy = np.max(img, 0).squeeze()
    return mip_xy

def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def normalize_percentile(im, low, high):
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


####################
# image convert
####################

def reverse_norm(im, mi, ma, scale=1, mode='8bit'):
    alpha = ma - mi
    beta = mi
    min_ = 0
    # min_ = np.percentile(im, 0.2)
    im = np.clip(im, 0, np.max(im))
    if mode == '8bit':
        if ma > 255:
            alpha = 255 / ma * beta
        im = (alpha * im + beta).astype(np.uint16)
        im = im.astype(np.uint8)
    elif mode == '16bit':
        # im = (im - min_) / (max_ - min_ + 1e-10) * 65535
        # im = im.astype(np.uint16)
        im = (scale * alpha * im + beta).astype(np.uint16)
    return im


def tensor2img(tensor, out_type=np.uint16, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # to range [0,1]
    tensor = tensor.squeeze().float().cpu()
    # tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    # if n_dim == 4:
    #     n_img = len(tensor)
    #     img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
    #     img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    if n_dim == 5:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 3, 0))  # HWC, BGR
    elif n_dim == 4:
        img_np = tensor.numpy()
        img_np = img_np[0, :]  # Gray
        # img_np = np.transpose(img_np, (1, 2, 3, 0))  # HWC, BGR
    elif n_dim == 3:
        # img_np = tensor.numpy()
        # img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        img_np = tensor.numpy()
        # img_np = img_np[0, :]  # Gray
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    # if out_type == np.uint16:
    #     img_np = reverse_norm(img_np)
    # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np


def save_img_t(img, mi, ma, img_path, scale=1, mode='8bit'):
    img = np.expand_dims(img, 1)
    img = reverse_norm(img, mi, ma, scale, mode)
    tifffile.imwrite(img_path, img, imagej=True)
    mip_xy = np.max(img, 0).squeeze()
    return mip_xy


def save_img_t(img, img_path, mode='8bit'):
    # img = np.expand_dims(img, 1)
    img = reverse_norm_t(img, mode)
    tifffile.imwrite(img_path, img)
    # tifffile.TiffWriter(file=img_path, bigtiff=False).write(img)
    # cv2.imwrite(img_path, img)


def reverse_norm_t(im, mode='8bit'):
    max_ = np.max(im)
    min_ = 0
    # min_ = np.percentile(im, 0.2)
    im = np.clip(im, min_, max_)
    if mode == '8bit':
        im = (im - min_) / (max_ - min_ + 1e-10) * 255
        im = im.astype(np.uint8)
    return im


####################
# metric
####################
def _cutoff(image, min_val, max_val):
    image = image.copy()
    image[image > max_val] = max_val
    image[image < min_val] = min_val
    return image


def calc_psnr_3d(ref_image, test_image, mask=None, data_range=[None, None]):
    """Calculates 3D PSNR.
    Args:
        ref_image (numpy.ndarray): The reference image.
        test_image (numpy.ndarray): The testing image.
        mask (numpy.ndarray): Calculate PSNR in this mask.
        data_range (iterable[float]): The range of possible values.

    Returns:
        float: The calculated PSNR.
    """
    mask = np.ones_like(ref_image) > 0 if mask is None else mask > 0
    min_val = np.min(ref_image) if data_range[0] is None else data_range[0]
    max_val = np.max(ref_image) if data_range[1] is None else data_range[1]

    ref_image = _cutoff(ref_image, min_val, max_val)
    test_image = _cutoff(test_image, min_val, max_val)

    mse = np.mean((ref_image[mask] - test_image[mask]) ** 2)
    psnr = 10 * np.log10((max_val - min_val) ** 2 / (mse + 1e-10))

    return psnr


def calc_psnr_3d_torch(ref_image, test_image, mask=None):
    """Calculates 3D PSNR.
    Args:
        ref_image (torch.Tensor): The reference image.
        test_image (torch.Tensor): The testing image.
        mask (torch.Tensor): Calculate PSNR in this mask.
        data_range (iterable[float]): The range of possible values.

    Returns:
        float: The calculated PSNR.
    """
    mask = torch.ones_like(ref_image) > 0 if mask is None else mask > 0
    min_val = torch.min(ref_image)
    max_val = torch.max(ref_image)

    mse = torch.mean((ref_image[mask] - test_image[mask]) ** 2)
    psnr = 10 * torch.log10((max_val - min_val) ** 2 / (mse + 1e-10))
    return psnr


class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()
def list_any_in_str(s_list, s):
    if isinstance(s_list, str):
        return s_list in s

    for s_item in s_list:
        if s_item in s:
            return True

    return False

def slice2str(s):
    if isinstance(s, slice):
        s = [s]

    return ",".join([f"{i.start}->{i.stop}" for i in s])

