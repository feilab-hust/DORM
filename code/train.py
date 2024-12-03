import os
import math
import argparse
import random
import logging
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.util import augment

import options.options as option
from utils import util
from models import create_model
from options.train.Config_train import chunking_data


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    model_name, GAN_model = chunking_data()
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    args.opt = 'options/train/train.yml'
    opt = option.parse(args.opt, is_train=True, model_name=model_name)
    cpu_threads_num = opt['cpu_threads_num']
    torch.set_num_threads(cpu_threads_num)

    #### distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + model_name, level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + model_name, level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('train_psnr', opt['path']['log'], 'train_psnr_' + model_name, level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in model_name:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + model_name)
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    file = 'data/training_data.npz'
    f = np.load(file, allow_pickle=True)
    X, Y, Z = f['X'], f['Y'], f['Z']  # GT, LR, MR

    data_size = len(X.shape)
    if data_size == 5 and GAN_model == '1':
        opt['model'] = 'GanModel'

    opt['net_dim'] = data_size

    # 20% data used for test
    test_data_split = int(len(X) * 0.2)
    train_data_split = len(Y) - test_data_split

    train_set = []
    test_set = []
    train_set.append(X[:train_data_split, :])
    train_set.append(Y[:train_data_split, :])
    test_set.append(X[train_data_split:, :])
    test_set.append(Y[train_data_split:, :])

    batch_size = opt['train']['batch_size']
    train_size = int(math.ceil(len(train_set) / batch_size))
    total_iters = int(opt['train']['niter'])
    total_epochs = int(math.ceil(total_iters / train_size))
    train_sampler = None

    if rank <= 0:
        logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
            len(train_set), train_size))
        logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
            total_epochs, total_iters))

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))

    psnr_best = 0
    for epoch in range(start_epoch, total_epochs + 1):
        avg_psnr_train = 0
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for iter in range(total_iters):
            current_step += 1
            if current_step > total_iters:
                break
            #### training

            random_number = random.sample(range(0, train_data_split), batch_size)

            img_LQ, img_GT = train_set[1][random_number], train_set[0][random_number]

            img_LQ, img_GT = augment([img_LQ, img_GT], True, True, )
            img_GT = torch.from_numpy(np.ascontiguousarray(img_GT)).float()
            img_LQ = torch.from_numpy(np.ascontiguousarray(img_LQ)).float()

            train_data = {'LQ': img_LQ, 'GT': img_GT}
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in model_name:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            ##### caculate train PSNR
            crop_size = opt['scale']
            bit = 65535.  # 255. for RGB 8bit
            gt_img = train_data['GT'].squeeze()

            sr_img = model.get_train_SR().squeeze()
            if gt_img.ndim == 3:
                gt_img = gt_img.unsqueeze(0)
                sr_img = sr_img.unsqueeze(0)
                cropped_sr_img = sr_img[:, :, crop_size:-crop_size, crop_size:-crop_size]
                cropped_gt_img = gt_img[:, :, crop_size:-crop_size, crop_size:-crop_size]
            else:
                cropped_sr_img = sr_img[:, crop_size:-crop_size, crop_size:-crop_size, crop_size:-crop_size]
                cropped_gt_img = gt_img[:, crop_size:-crop_size, crop_size:-crop_size, crop_size:-crop_size]
            avg_psnr_train += util.calc_psnr_3d_torch(cropped_sr_img * int(bit), cropped_gt_img * int(bit))
            # avg_psnr_train = 0
            # validation
            save_test_data_num = 4
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
                avg_psnr = 0.0
                idx = 0
                for i in range(test_data_split):

                    # img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    img_LQ, img_GT = test_set[1][idx], test_set[0][idx]

                    img_GT = torch.from_numpy(np.ascontiguousarray(np.expand_dims(img_GT, axis=0))).float()
                    img_LQ = torch.from_numpy(np.ascontiguousarray(np.expand_dims(img_LQ, axis=0))).float()

                    val_data = {'LQ': img_LQ, 'GT': img_GT}

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()

                    gt_img = util.tensor2img(visuals['GT'])  # uint16
                    sr_img = util.tensor2img(visuals['SR'])  # uint16
                    lr_img = util.tensor2img(visuals['LR'])  # uint16
                    if i < save_test_data_num:
                        img_name = str(idx)
                        img_dir = os.path.join(opt['path']['val_images'], img_name)
                        util.mkdir(img_dir)
                        save_img_path = os.path.join(img_dir, '{:s}_SR_{:d}.tif'.format(img_name, current_step))
                        util.save_img_t(sr_img, save_img_path)
                        # Save ground truth

                        if current_step == opt['train']['val_freq']:
                            save_img_path_gt = os.path.join(img_dir, '{:s}_GT_{:d}.tif'.format(img_name, current_step))
                            util.save_img_t(gt_img, save_img_path_gt)
                            save_img_path_L = os.path.join(img_dir, '{:s}_LR_{:d}.tif'.format(img_name, current_step))
                            util.save_img_t(lr_img, save_img_path_L)

                    # calculate PSNR

                    crop_size = opt['scale']
                    bit = 65535.  # 255. for RGB 8bit
                    gt_img = gt_img / bit
                    sr_img = sr_img / bit
                    if gt_img.ndim == 2:
                        gt_img = np.expand_dims(gt_img, axis=0)
                        sr_img = np.expand_dims(sr_img, axis=0)
                        cropped_sr_img = sr_img[:, crop_size:-crop_size, crop_size:-crop_size]
                        cropped_gt_img = gt_img[:, crop_size:-crop_size, crop_size:-crop_size]
                    else:
                        cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, crop_size:-crop_size]
                        cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, crop_size:-crop_size]
                    # avg_psnr += util.calculate_psnr(cropped_sr_img * int(bit), cropped_gt_img * int(bit), bit)
                    avg_psnr += util.calc_psnr_3d(cropped_sr_img * int(bit), cropped_gt_img * int(bit))
                    idx += 1

                avg_psnr = avg_psnr / idx

                # log
                logger.info('# Validation # PSNR: {:.4e}.'.format(avg_psnr))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}.'.format(
                    epoch, current_step, avg_psnr))
                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in model_name:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)
                if avg_psnr >= psnr_best:
                    psnr_best = avg_psnr
                    model.save('best')

                #### log
            if iter % train_data_split == 0:
                avg_psnr_train /= train_size
                logger.info('# Train # PSNR: {:.4e}.'.format(avg_psnr_train))
                logger_tarin_psnr = logging.getLogger('train_psnr')  # validation logger
                logger_tarin_psnr.info('<epoch:{:3d}, iter:{:8,d}> train_psnr: {:.4e}.'.format(
                    epoch, current_step, avg_psnr_train))
                avg_psnr_train = 0
                if opt['use_tb_logger'] and 'debug' not in model_name:
                    tb_logger.add_scalar('psnr', avg_psnr_train, current_step)
                    avg_psnr_train = 0



    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()
