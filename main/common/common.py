import logging
import torch as t

from main.conf.configuration import Configuration

#
conf = Configuration()


def cuda(tensor, device=None):
    if device is None:
        device = conf.get('device')
        if device is None:
            device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        else:
            device = t.device(device)

    return tensor.to(device)


def getLogger(self):
    """

    :rtype:
    """
    return logging.getLogger(self.__class__.__name__)
