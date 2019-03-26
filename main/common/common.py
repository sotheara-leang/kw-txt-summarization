import logging
import torch as t

from main.conf.configuration import Configuration


#
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('main')

#
conf = Configuration()


#
def cuda(tensor):
    if t.cuda.is_available():
        tensor = tensor.cuda()
    return tensor
