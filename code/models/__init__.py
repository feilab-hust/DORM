import logging

logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    if model == 'SingleModel':
        from .SingleModel import SingleModel as M
    elif model == 'IR_new':
        from .image_restoration_model_improve import ImageRestorationModel as M
    elif model == 'GanModel':
        from .GanModel import GanModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
