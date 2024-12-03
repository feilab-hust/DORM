import numpy as np
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import imageio


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
    return im.astype(np.float32)


class LQDataset(data.Dataset):
    '''Read LQ images only in the test phase.'''

    def __init__(self, opt):
        super(LQDataset, self).__init__()
        self.opt = opt
        self.is_train = self.opt['is_train']
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.sizes_LQ = None, None
        self.LQ_env = None  # environment for lmdb
        self.low_p = self.opt['val']['low_p']
        self.high_p = self.opt['val']['high_p']
        self.xy_clear_border = self.opt['val'].get('xy_clear_border', 0)
        self.mi = 0.0
        self.ma = 0.0

        # read image list from lmdb or image files
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(opt['data_type'], opt['dataroot_LQ'])
        assert self.paths_LQ, 'Error: LQ paths are empty.'

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb' and self.LQ_env is None:
            self._init_lmdb()

        if self.is_train:
            LQ_path = None

            # get LQ image
            LQ_path = self.paths_LQ[index]
            if self.data_type == 'lmdb':
                resolution = [int(s) for s in self.sizes_LQ[index].split('_')]
            else:
                resolution = None
            img_LQ = util.read_img(self.LQ_env, LQ_path, resolution).squeeze()[:, :, :, np.newaxis]

            img_LQ = torch.from_numpy(
                np.ascontiguousarray(np.transpose(normalize_percentile(img_LQ, 0, 100), (3, 0, 1, 2)))).float()

            return {'LQ': img_LQ, 'LQ_path': LQ_path}

        else:
            LQ_path = self.paths_LQ[index]
            img = imageio.volread(LQ_path)
            if self.xy_clear_border:
                img = img[:, 0:-self.xy_clear_border, self.xy_clear_border:]
            self.mi, self.ma = np.percentile(img, self.low_p), np.percentile(img, self.high_p)
            img = normalize_min_max(img, max_v=self.ma, min_v=self.mi)
            img = torch.from_numpy(img)

            return img, self.mi, self.ma


    def __len__(self):
        return len(self.paths_LQ)
