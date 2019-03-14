import logging

from main.conf.configuration import Configuration


#
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#
conf = Configuration()
