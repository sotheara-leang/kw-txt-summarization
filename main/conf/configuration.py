import yaml
import os
import logging

import main.common.logging

from yaml import Loader
from singleton_decorator import singleton


@singleton
class Configuration:

    def __init__(self):
        logging.debug('init config.yml')

        with open(os.path.join(os.path.dirname(__file__), "config.yml"), 'r') as file:
            self.cfg = yaml.load(file, Loader=Loader)

    def get(self, key):
        return self.cfg[key]
