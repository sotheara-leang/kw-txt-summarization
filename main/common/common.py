import logging
import os
import random

import numpy as np
import torch as t
from singleton_decorator import singleton

from main.common.configuration import Configuration
from main.common.logger import Logger

ctx = globals()

@singleton
class AppContext(object):

    def init_rand_seed(self):
        random.seed(123)
        np.random.seed(123)
        t.manual_seed(123)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(123)

    def __init__(self, conf_file=None):
        self.init_rand_seed()

        if conf_file is None:
            raise Exception('Configuration file not found')

        self.conf = Configuration(conf_file)

        if self.conf.get('logging:enable') is True:
            log_file = self.conf.get('logging:conf-file')
            if log_file is None:
                log_file = os.path.dirname(conf_file) + '/logging.yml'

            Logger(log_file)
        else:
            logging.basicConfig(level=logging.DEBUG)

        ctx['conf'] = self.conf

def cuda(tensor, device=None):
    if device is None:
        device = ctx['conf'].get('device')
        if device is None:
            device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        else:
            device = t.device(device)

    return tensor.to(device)

def logger(self):
    return logging.getLogger(self.__class__.__name__)

def conf(key=None, default=None):
    if key is None:
        return ctx['conf']
    return ctx['conf'].get(key, default)
