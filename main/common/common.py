import logging
import torch as t

from main.conf.configuration import Configuration


#
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('main')

#
conf = Configuration()

#
device = t.device('cuda' if t.cuda.is_available() else 'cpu')