import os.path
import sys
import time

import torch
from utils.logger import get_logger
from utils.util import list_any_in_str, slice2str
from utils.files import mkdirs
from options import options
from data import create_dataset, create_dataloader
from models import create_model
import numpy as np
import threading
import tifffile
import imageio
from scipy.ndimage.interpolation import zoom

logger = get_logger("inference_single_stage")


def predict_tile(model, x, net_axes_in_div_by, block_size, overlap, method="cover", net_dim=5):
    def __broadcast(val):
        val = [val for i in x.shape] if np.isscalar(val) else val
        return val

    def _split(v):
        a = v // 2
        return a, v - a

    def slice_generator(x_pad, block_img_size, overlap, factor=1):
        in_slice_list, out_slice_list, crop_slice_list = [], [], []
        anchors = [(z, y, x)
                   for z in range(overlap[0], x_pad.shape[0], block_img_size[0])
                   for y in range(overlap[1], x_pad.shape[1], block_img_size[1])
                   for x in range(overlap[2], x_pad.shape[2], block_img_size[2])]

        for i, anchor in enumerate(anchors):
            begin = [p - o for p, o in zip(anchor, overlap)]  # 保证从0开始， 并且带overlap
            end = [p + b + o for p, b, o in zip(anchor, block_img_size, overlap)]
            if not all(i <= j for i, j in zip(end, x_pad.shape)):
                continue

            in_slice_list.append(tuple([slice(b, e) for b, e in zip(begin, end)]))
            # crop和out部分只取中间部分，舍弃无用pad以及边缘像素
            out_slice_list.append(
                tuple([slice((b + o // 2) * factor, (e - o // 2) * factor) for b, e, o, in zip(begin, end, overlap)]))
            crop_slice_list.append(
                tuple([slice((c // 2) * factor, (b + c + c // 2) * factor) for b, c in zip(block_img_size, overlap)]))

        # for i in range(len(in_slice_list)):
        #     logger.info(f"in slice list: {in_slice_list[i]}, out slice list: {out_slice_list[i]}, crop slice list: {crop_slice_list[i]}")

        return in_slice_list, out_slice_list, crop_slice_list

    assert method in ["mean", "cover"], "invalid method"

    axes = list(range(x.ndim))
    block_size = __broadcast(block_size)
    overlap = __broadcast(overlap)
    net_axes_in_div_by = __broadcast(net_axes_in_div_by)

    block_img_size = [r // b + 2 * o if b > 1 else r for r, b, o in zip(x.shape, block_size, overlap)]
    block_img_size = [int(d * np.ceil(b / d)) for b, d in zip(block_img_size, net_axes_in_div_by)]

    # check overlap valid
    overlap = [i if i > 1 else i * s for i, s in zip(overlap, block_img_size)]
    overlap = [0 if b >= s else i for i, b, s in zip(overlap, block_img_size, x.shape)]
    overlap = [i if i % 2 == 0 else i + 1 for i in overlap]

    block_img_size = [int(b - 2 * i) for b, i in zip(block_img_size, overlap)]

    im_size_padded = [((g + 1) * b if o != 0 else g * b) for g, b, o in zip(block_size, block_img_size, overlap)]

    def pad_list_to_valid(input_list, target_shape):
        for idx, (value, target) in enumerate(zip(input_list, target_shape)):
            if value < target:
                input_list[idx] = (target + 7) // 8 * 8

        return input_list

    im_size_padded = pad_list_to_valid(im_size_padded, [i + o // 2 for o, i in zip(overlap, x.shape)])

    pad = {
        a: (o // 2, k - i - o // 2)
        for a, o, i, k in zip(axes, overlap, x.shape, im_size_padded)
    }

    crop = tuple(
        slice(p[0] * model.factor, -p[1] * model.factor if p[1] > 0 else None)
        for p in (pad[a] for a in axes)
    )

    x_pad = np.pad(x, tuple(pad[a] for a in axes), mode="reflect")
    logger.info(f"Pad img {x.shape} => {x_pad.shape} with pad: {pad}")
    sr_size = [i * model.factor for i in x_pad.shape]
    ret = np.zeros(sr_size, dtype=np.float32)
    if method == "mean":
        count = np.zeros(sr_size, dtype=np.int32)
    if net_dim == 5:
        input_slice, output_slice, crop_slice = slice_generator(x_pad, block_img_size, overlap, model.factor)

        logger.info(f"predict start with {len(input_slice)} patches...")
        for pdx, (i_slice, o_slice, crop_slice) in enumerate(zip(input_slice, output_slice, crop_slice)):
            patch_start = time.time()

            logger.debug(
                f"get patch: i_slice: {slice2str(i_slice)}, o_slice: {slice2str(o_slice)}, crop_slice: {slice2str(crop_slice)}")
            block = x_pad[i_slice]
            private_method = f"model._{model.__class__.__name__}__predict_block"
            out = eval(private_method)(block)

            # linear blending
            for i_a in range(x.ndim):
                if i_slice[i_a].start != 0:
                    blending_axes = i_a
                    logger.debug(f"blending axes: {blending_axes}")
                    for i in range(overlap[blending_axes] * model.factor):
                        src_weight = (i + 1) / (overlap[blending_axes] * model.factor + 1)
                        dst_weight = 1 - src_weight
                        blending_slice = tuple(
                            [crop_slice[j] if j != blending_axes else crop_slice[j].start + i for j in range(x.ndim)])
                        ret_slice = tuple(
                            [o_slice[j] if j != blending_axes else o_slice[j].start + i for j in range(x.ndim)])
                        out[blending_slice] = src_weight * out[blending_slice] + dst_weight * ret[ret_slice]

            if method == "mean":
                count[o_slice] += 1
                ret[o_slice] += out[crop_slice]
            else:
                ret[o_slice] = out[crop_slice]

            patch_end = time.time()
            patch_time = patch_end - patch_start
            logger.info(f"patch [{pdx + 1:02d}]/ [{len(input_slice)}] shape: {block.shape} time: {patch_time:.4f}s")
    else:
        d, h, w = x_pad.shape
        private_method = f"model._{model.__class__.__name__}__predict_block"

        ret_1 = np.zeros((d, h, w))
        ret_2 = np.zeros((d, h, w))
        for i in range(w):
            ret_1[:, :, i] = eval(private_method)(x_pad[:, :, i])
        for j in range(w):
            ret_2[:, j, :] = eval(private_method)(x_pad[:, j, :])
        ret = (np.sqrt(np.maximum(ret_1, 0) * np.maximum(ret_2, 0)))

    # torch.cuda.empty_cache()
    if method == "mean":
        return ret[crop] / count[crop]
    else:
        return ret[crop]


def reverse_and_save_img(img, mi, ma, save_img_path, save_mip_path, scale=1, mode="16bit"):
    if img is None:
        return

    if img.ndim == 3:
        img = np.expand_dims(img, 1)

    alpha = ma - mi
    beta = mi

    if mode == "8bit":
        if ma > 255:
            alpha = 255 / ma * beta

        img = alpha * img + beta
        img = img.astype(np.uint8)
    elif mode == "16bit":
        img = np.clip(img, 0, 65535)
        img = scale * alpha * img + beta
        img = img.astype(np.uint16)

    tifffile.imwrite(save_img_path, img, imagej=True)

    if save_mip_path:
        mip_xy = np.max(img, 0).squeeze()
        imageio.imsave(save_mip_path, mip_xy)

    return True


def main(opt):
    torch.cuda.empty_cache()
    model = create_model(opt)

    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_set_name = test_loader.dataset.opt['name']
        logger.info('Testing [{:s}]...'.format(test_set_name))

        load_data_start = time.time()
        predict_time_start = time.time()
        for idx, (test_data, mi, ma) in enumerate(test_loader):
            # if idx > 8:
            tiff_time_start = time.time()
            tiff_file_path = test_set.paths_LQ[idx]
            test_img = test_data[0].numpy()
            if dataset_opt['zoom_scale'] and dataset_opt['zoom_scale'] > 1:
                test_img = zoom(test_img, (dataset_opt['zoom_scale'], 1, 1), order=1)
            mi = mi[0].numpy()
            ma = ma[0].numpy()
            logger.info(f"Processing [{idx + 1}]/[{len(test_loader)}]: {tiff_file_path}...")

            load_data_end = time.time()
            load_data_time = load_data_end - load_data_start
            logger.info(f"Loading data and percentile(mi: {mi}, ma: {ma}) time: {load_data_time:.4f}s")

            net_axes_in_div_by = [opt['val'].get('min_devide_by', 8)] * test_img.ndim
            block_size, overlap = model.block_size, model.over_lap

            net_dim = opt['net_dim']
            result = predict_tile(model, test_img, net_axes_in_div_by, block_size, overlap, net_dim=net_dim)

            model_name = model.opt['path']['pretrain_model_C'].split('\\')[-3]
            img_name = model.opt['path']['pretrain_model_C'].split('\\')[-1].split('.')[0][:-2] + '_SR_' + \
                       os.path.split(tiff_file_path)[-1]
            base_dir = test_loader.dataset.opt['dataroot_LQ']
            save_img_dir = os.path.join(base_dir, model_name)
            mkdirs(save_img_dir)
            save_img_path = os.path.join(save_img_dir, img_name)
            save_mip_path = ""

            logger.info(f"result save img path: {save_img_path}")
            reverse_and_save_img(result, mi, ma, save_img_path, save_mip_path, model.factor, model.mode)

            load_data_start = time.time()
            tiff_time_end = time.time()
            logger.info(f"Tiff completed! time: {tiff_time_end - tiff_time_start:.4f}s")

        predict_time_end = time.time()
        logger.info(f"Dataset [{test_set_name}] completed! time: {predict_time_end - predict_time_start:.4f}s")


if __name__ == "__main__":
    import argparse

    config_yaml_paths = ["options/test/test_CropPatch.yml"]

    for config_yaml_path in config_yaml_paths:
        parser = argparse.ArgumentParser()
        parser.add_argument('-opt', type=str, default=config_yaml_path, help='Path to options YMAL file.')
        opt = options.parse(parser.parse_args().opt, is_train=False)
        opt = options.dict_to_nonedict(opt)

        logger.info(f"Read data/model config from yaml file: {config_yaml_path}")
        logger.info("Yaml params: \n" + options.dict2str(opt))

        for key, path in opt["path"].items():
            if key != "experiments_root" and not list_any_in_str(["pretrain_model", "resume"], key):
                if not os.path.exists(path):
                    mkdirs(path)
                    logger.info(f"Create {key} path: {path}")

        main(opt)
