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
    #
    # def __normalize_percentile(self, im, low=1, high=99.99):
    #     # def __normalize_percentile(self, im, low=0.2, high=99.5):
    #     return normalize_percentile(im.astype(self.dtype), low=low, high=high)
    #
    # def __normalize(self, im, max_v):
    #     im = im.astype(self.dtype)
    #     return normalize_min_max(im, max_v)
    #
    # def __reverse_norm(self, im, max_raw=65535):
    #     max_ = np.max(im)
    #     min_ = np.percentile(im, 0.2)
    #     # min_ = np.min(im)
    #     im = np.clip(im, min_, max_)
    #     im = (im - min_) / (max_ - min_) * max_raw
    #     return im.astype(np.uint16)
    #
    # def __check_dims(self, im, block_size, overlap):
    #     # To-Do: deal with float block_size and overlap
    #     def __broadcast(val):
    #         val = [val for i in im.shape] if np.isscalar(val) else val
    #         return val
    #
    #     def __check_block_size(block_size, di_int=8):
    #         block_shape_new = [int(8 * np.ceil(b / di_int)) for b in block_size]
    #         return block_shape_new
    #
    #     assert len(im.shape) == 3 or len(im.shape) == 2, 'Error:the input image must be in shape [depth, height, width]'
    #
    #     block_size = [r // b + 2 * overlap if b > 1 else r for r, b in zip(im.shape, block_size)]
    #     block_size = __check_block_size(block_size)
    #     block_size = __broadcast(block_size)
    #     overlap = __broadcast(overlap)
    #     assert len(block_size) == len(
    #         im.shape), 'Error:ndim of block_size ({}) mismatch that of image size ({})'.format(block_size, im.shape)
    #     assert len(overlap) == len(im.shape), 'Error:ndim of overlap ({}) mismatch that of image size ({})'.format(
    #         overlap, im.shape)
    #
    #     # block_size = [b if b <= i else i for b, i in zip(block_size, im.shape)]
    #
    #     overlap = [i if i > 1 else i * s for i, s in zip(overlap, block_size)]
    #     overlap = [0 if b >= s else i for i, b, s in zip(overlap, block_size,
    #                                                      im.shape)]  # no overlap along the dims where the image size equal to the block size
    #     overlap = [i if i % 2 == 0 else i + 1 for i in overlap]  # overlap must be even number
    #
    #     block_size = [b - 2 * i for b, i in zip(block_size, overlap)]  # real block size when inference
    #
    #     overlap = [int(i) for i in overlap]
    #     block_size = [int(i) for i in block_size]
    #     logger.info('block size (overlap excluded) : {} overlap : {}'.format(block_size, overlap))
    #
    #     return block_size, overlap
    #
    # def _padding_block(self, im, blk_size, overlap):
    #     grid_dim = [int(np.ceil(float(i) / b)) for i, o, b in zip(im.shape, overlap, blk_size)]
    #     im_size_padded = [(g * b + b if o != 0 else g * b) for g, b, o in zip(grid_dim, blk_size, overlap)]
    #
    #     # im_wrapped = np.ones(im_size_padded, dtype=self.dtype) * np.min(im)
    #     # im_wrapped = np.ones(im_size_padded, dtype=self.dtype) * np.percentile(im, 1)
    #
    #     valid_region = [slice(o // 2, o // 2 + i) for o, i in zip(overlap, im.shape)]
    #     padding = tuple([(o // 2, k - i - o // 2) for o, i, k in zip(overlap, im.shape, im_size_padded)])
    #
    #     im_wrapped = np.pad(im, padding, mode='reflect')
    #     # tifffile.imwrite(r'K:\livingcell\561\test.tif', im_wrapped)
    #
    #     sr_valid_region = [slice(o // 2 * self.factor, (o // 2 + i) * self.factor) for o, i in zip(overlap, im.shape)]
    #     logger.info('raw image size : {}, wrapped into : {}'.format(im.shape, im_size_padded))
    #     logger.info('valid region index: {} 【{} after SR】'.format(valid_region, sr_valid_region))
    #
    #     # im_wrapped[tuple(valid_region)] = im
    #
    #     return im_wrapped, sr_valid_region
    #
    # def __region_iter(self, im, blk_size, overlap, factor):
    #     """
    #     Params:
    #         -im: ndarray in dims of [depth, height, width]
    #     """
    #     im_size = im.shape
    #
    #     anchors = [(z, y, x)
    #                for z in range(overlap[0], im_size[0], blk_size[0])
    #                for y in range(overlap[1], im_size[1], blk_size[1])
    #                for x in range(overlap[2], im_size[2], blk_size[2])]
    #
    #     for i, anchor in enumerate(anchors):
    #         # revised_overlap = [0 if a == i else i for a, i in zip(anchor, overlap)]
    #         begin = [p - c for p, c in zip(anchor, overlap)]
    #         end = [p + b + c for p, b, c in zip(anchor, blk_size, overlap)]
    #         yield [slice(b, e) for b, e in zip(begin, end)], \
    #             [slice((b + c // 2) * factor, (e - c // 2) * factor) for b, e, c, in zip(begin, end, overlap)], \
    #             [slice((c // 2) * factor, (b + c + c // 2) * factor) for b, c in zip(blk_size, overlap)]
    #
    # def GaussianBlur2d(self, x, sigma_x=4.):
    #     def gaussian(window_size, sigma):
    #         gauss = np.array([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    #         return gauss / gauss.sum()
    #
    #     def create_kernel(kernel_size=3, sigma_x=1., sigma_y=1., sigma_z=1.):
    #         x_1D_window = gaussian(kernel_size, sigma_x)[:, np.newaxis]
    #         y_1D_window = gaussian(kernel_size, sigma_y)[np.newaxis, :]
    #         xy_2D_window = np.multiply(x_1D_window, y_1D_window)
    #
    #         kernel = xy_2D_window
    #         return kernel
    #
    #     kernel_size = int(sigma_x * 2) + 1
    #     kernel = create_kernel(kernel_size, sigma_x, sigma_x)
    #     from scipy.signal import fftconvolve
    #     x = fftconvolve(x, kernel, mode='same')
    #     return x
    #
    # def predict_without_norm(self, im, block_size, overlap):
    #
    #     block_size, overlap = self.__check_dims(im, block_size, overlap)
    #     factor = self.factor
    #
    #     im_wrapped, valid_region_idx = self._padding_block(im, block_size, overlap)
    #     sr_size = [s * factor for s in im_wrapped.shape]
    #     sr = np.zeros(sr_size, dtype=self.dtype)
    #     slice_list = list(self.__region_iter(im_wrapped, block_size, overlap, factor))
    #
    #
    #     for idx, (src, dst, in_blk) in enumerate(slice_list):
    #         # print('source: {}  dst: {}  valid: {} '.format(src, dst, in_blk))
    #         begin = [i.start for i in src]
    #         end = [i.stop for i in src]
    #
    #         im_th = 0.1
    #
    #         if not all(i <= j for i, j in zip(end, im_wrapped.shape)):
    #             continue
    #
    #         # print('\revaluating {}-{} in {}  '.format(begin, end, im_wrapped.shape), end='')
    #         print('evaluating {}-{} -> out slice: {}, crop slice: {} '.format(begin, end, dst, in_blk))
    #         block = im_wrapped[tuple(src)]
    #
    #         # block_blur = self.GaussianBlur2d(np.max(block, axis=0))
    #         predict_start = time.time()
    #         block = self.__predict_block(block)
    #         predict_end = time.time()
    #         logger.info(f"predict [{idx + 1}]/[{len(slice_list)}] elapsed {predict_end - predict_start:.04f}")
    #         sr[tuple(dst)] = block[tuple(in_blk)]
    #
    #     print('')
    #     return sr[tuple(valid_region_idx)]
    #
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

    # def predict(self, im, normalization='auto', **kwargs):
    #     block_size = self.block_size
    #     overlap = self.over_lap
    #     assert normalization in ['fixed', 'auto']
    #     norm_fn = self.__normalize if normalization is 'fixed' else self.__normalize_percentile
    #     im = norm_fn(im, **kwargs)
    #
    #     logger.info('normalized to [%.4f, %.4f]' % (np.min(im), np.max(im)))
    #     sr = self.predict_without_norm(im, block_size, overlap)
    #     torch.cuda.empty_cache()
    #     # sr =self.__reverse_norm(sr, im_max)
    #     return sr

    # def dist_validation(self, dataloader, dataset_dir):
    #     def get_file_list(path_, regx):
    #         file_list = os.listdir(path_)
    #         file_list = [f for _, f in enumerate(file_list) if re.search(regx, f)]
    #         return file_list
    #
    #     base_dir = dataloader.dataset.opt['dataroot_LQ']
    #     valid_lr_imgs = get_file_list(path_=base_dir, regx='.*.tif')
    #
    #     for im_idx, im_file in enumerate(valid_lr_imgs):
    #         # if im_idx > 24:
    #         logger.info(f"Processing:[{im_idx}]/[{len(valid_lr_imgs)}]: {im_file}")
    #         time_begin = time.time()
    #         temp_time_start = time.time()
    #         img = imageio.volread(os.path.join(base_dir, im_file))[:, 0:-10, 10:]
    #         temp_time_end = time.time()
    #         read_time = temp_time_end - temp_time_start
    #         logger.info(f"Read tiff time: {temp_time_end - temp_time_start:.4f}s")
    #
    #         temp_time_start = time.time()
    #         mi = np.percentile(img, self.low_p)
    #         ma = np.percentile(img, self.high_p)
    #         temp_time_end = time.time()
    #         percentile_time = temp_time_end - temp_time_start
    #         logger.info(f"Percentile low({self.low_p})={mi}, high({self.high_p})={ma} time: {temp_time_end - temp_time_start:.4f}s")
    #         # img_min = np.percentile(np.max(img, axis=0), 0.5)
    #         # img_min = 90
    #         # bg = np.random.normal(img_min, img_min/30, size=img.shape).astype(np.float32)
    #         # img = np.where(img <= img_min, bg, img)
    #         # img = img[12:44, :, :]
    #         model_name = self.opt['path']['pretrain_model_C'].split('\\')[-3]
    #         img_name = self.opt['path']['pretrain_model_C'].split('\\')[-1].split('.')[0][:-2] + '_SR_' + im_file
    #         temp_time_start = time.time()
    #         sr_img = self.predict(img, low=self.low_p, high=self.high_p)
    #         temp_time_end = time.time()
    #         predict_time = temp_time_end - temp_time_start
    #         import scipy.ndimage as ndimage
    #         # stack1 = ndimage.grey_erosion(sr_img, size=50)
    #         # sr_img = sr_img - stack1
    #
    #         # sr_img = zoom(sr_img, 55 / 60, order=3)
    #         temp_time_start = time.time()
    #         save_img_path = osp.join(base_dir, model_name)
    #         if not os.path.exists(save_img_path):
    #             os.makedirs(save_img_path)
    #         save_img_path = osp.join(save_img_path, img_name)
    #         save_img(sr_img, mi, ma, save_img_path, scale=self.factor, mode='16bit')
    #         temp_time_end = time.time()
    #         save_time = temp_time_end - temp_time_start
    #         logger.info(f"Save img to {save_img_path} time: {temp_time_end - temp_time_start:.4f}s")
    #         time_end = time.time()
    #         logger.info(f"Processing time: {time_end - time_begin:.4f}s, read time: {read_time:.4f}s, percentile time: {percentile_time:.4f}s, predict time: {predict_time:.4f}s, save time: {save_time:.4f}")
    #
    # def nondist_validation(self, *args, **kwargs):
    #     logger = get_logger()
    #     logger.warning('nondist_validation is not implemented. Run dist_validation.')
    #     self.dist_validation(*args, **kwargs)
    #
    # def _log_validation_metric_values(self, current_iter, dataset_name,
    #                                   tb_logger, metric_dict):
    #     log_str = f'Validation {dataset_name}, \t'
    #     for metric, value in metric_dict.items():
    #         log_str += f'\t # {metric}: {value:.4f}'
    #     logger = get_root_logger()
    #     logger.info(log_str)
    #
    #     log_dict = OrderedDict()
    #     # for name, value in loss_dict.items():
    #     for metric, value in metric_dict.items():
    #         log_dict[f'm_{metric}'] = value
    #
    #     self.log_dict = log_dict
    #
    # def get_current_visuals(self):
    #     out_dict = OrderedDict()
    #     out_dict['lq'] = self.lq.detach().cpu()
    #     out_dict['result'] = self.output.detach().cpu()
    #     return out_dict
    #
    # def save(self, epoch, current_iter):
    #     self.save_network(self.net_g, 'net_g', current_iter)
    #     self.save_training_state(epoch, current_iter)
