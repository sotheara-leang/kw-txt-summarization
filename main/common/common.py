import logging
import torch as t

from singleton_decorator import singleton
from main.common.configuration import Configuration
from main.common.logger import Logger


ctx = globals()

@singleton
class AppContext(object):

    def __init__(self, conf_file=None):
        if conf_file is None:
            conf_file = 'main/conf/config.yml'

        self.conf = Configuration(conf_file)

        if self.conf.get('logging:enable') is True:
            log_dir = self.conf.get('logging:conf-file', 'main/conf/logging.yml')
            Logger(log_dir)
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
