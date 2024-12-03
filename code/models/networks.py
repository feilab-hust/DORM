import torch
import logging
# import codes.models.modules.discriminator_vgg_arch as SRGAN_arch
# from codes.models.modules.Inv_arch import *
# from codes.models.modules.Subnet_constructor import subnet
import models.modules.discriminator_vgg_arch as SRGAN_arch
from models.modules.Unet3d import Unet
from models.modules.Unet2d import Unet2d

import math

logger = logging.getLogger('base')


####################
# define network
####################
def define_C(opt):
    # scale = opt['scale']
    opt_net = opt['network_C']
    which_model = opt_net['which_model_C']
    if which_model == 'Unet':
        if opt['net_dim'] == 5:
            netC = Unet(num_features=opt_net['n_fea'], uptype=opt_net['uptype'])
        elif opt['net_dim'] == 4:
            netC = Unet2d(num_features=opt_net['n_fea'], uptype=opt_net['uptype'])
        else:
            raise NotImplementedError(
                'The expected dimensions of the original data are 5 or 4, but the provided data dimensions are[{:s}]'.format(
                    opt['net_dim']))
    else:
        raise NotImplementedError('Model [{:s}] not recognized'.format(which_model))
    return netC


# Discriminator
def define_D_HR(opt):
    opt_net = opt['network_D_HR']
    which_model = opt_net['which_model_D_HR']
    if which_model == 'discriminator_vgg_128':
        netD_HR = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'Unet_discriminator_sn':
        netD_HR = SRGAN_arch.UNetDiscriminatorSN(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'NLayerDiscriminator':
        netD_HR = SRGAN_arch.NLayerDiscriminator(in_nc=opt_net['in_nc'], nf=opt_net['nf'], n_layers=opt_net['n_layers'],
                                                 kw=opt_net['kw'])
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD_HR
