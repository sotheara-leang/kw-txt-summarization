import logging
import torch as t

from singleton_decorator import singleton

from main.common.configuration import Configuration


@singleton
class Bootstrap(object):

    def __init__(self):
        self.conf = Configuration('main/conf/config.yml', 'main/conf/logging.yml')


def cuda(tensor, device=None):
    if device is None:
        device = conf.get('device')
        if device is None:
            device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        else:
            device = t.device(device)

    return tensor.to(device)


def getLogger(self):
    return logging.getLogger(self.__class__.__name__)



bootstrap   = Bootstrap()
conf        = bootstrap.conf