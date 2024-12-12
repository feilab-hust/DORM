# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import logging

import tifffile
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
from models.base_model import BaseModel
import models.networks as networks
from torch.nn.parallel import DataParallel, DistributedDataParallel
from utils.logger import get_logger
from utils.util import save_img, tensor2img
from utils.dist_util import get_dist_info
import time
import os
import re
import imageio
import numpy as np
from scipy.ndimage.interpolation import zoom

logger = get_logger("model_improve")


def normalize_percentile(im, low, high):
    """Normalize the input 'im' by im = (im - p_low) / (p_high - p_low), where p_low/p_high is the 'low'th/'high'th percentile of the im
    """
    p_low, p_high = np.percentile(im, low), np.percentile(im, high)
    return normalize_min_max(im, max_v=p_high, min_v=p_low)


def normalize_min_max(im, max_v, min_v=0):
    eps = 1e-10
    try:
        import numexpr
        im = numexpr.evaluate("(im - min_v) / (max_v - min_v + eps)")
    except ImportError:
        im = (im - min_v) / (max_v - min_v + eps)
    return im


class ImageRestorationModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel, self).__init__(opt)

        # define network
        self.netC = networks.define_C(opt).to(self.device)
        if opt['dist']:
            self.netC = DistributedDataParallel(self.netC, device_ids=[torch.cuda.current_device()])
        else:
            self.netC = DataParallel(self.netC)
        # load pretrained models
        load_path_C = self.opt['path']['pretrain_model_C']
        if load_path_C is not None:
            logger.info('Loading model for C [{:s}] ...'.format(load_path_C))
            self.load_network(load_path_C, self.netC, self.opt['path']['strict_load'])

        if self.is_train:
            self.init_training_settings()

        crop_size_d = self.opt['val']['crop_size_d']
        crop_size_h = self.opt['val']['crop_size_h']
        crop_size_w = self.opt['val']['crop_size_w']
        over_lap = self.opt['val']['over_lap']
        self.low_p = self.opt['val']['low_p']
        self.high_p = self.opt['val']['high_p']
        self.mode = self.opt['val']['mode']

        self.factor = int(opt['scale'])
        self.block_size = (crop_size_d, crop_size_h, crop_size_w)
        self.over_lap = over_lap
        self.dtype = np.float32


    def feed_data(self, data):
        self.lq = data['LQ'].to(self.device)  # LQ
    
    def __predict_block(self, block):
        torch.backends.cudnn.benchmark = False
        b_shape = block.shape
        block = block[np.newaxis, np.newaxis, :]
        block = torch.from_numpy(block).float()
        block = block.cuda()
        self.netC.eval()
        with torch.no_grad():
            net_out = self.netC(block).squeeze()
            out = net_out.float().cpu().numpy()
            del net_out
            del block
            return out
