import yaml
import os

from yaml import Loader
from singleton_decorator import singleton


@singleton
class Configuration:

    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), "config.yml"), 'r') as file:
            self.cfg = yaml.load(file, Loader=Loader)

    def get(self, key):
        return self.cfg[key]

    def set(self, key, value):
        self.cfg.__setitem__(key, value)
