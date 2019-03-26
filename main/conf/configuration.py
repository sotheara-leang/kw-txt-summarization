import yaml
import os

from yaml import Loader, Dumper
from singleton_decorator import singleton
from main.common.util.file_util import FileUtil


@singleton
class Configuration:

    def __init__(self):
        with open(FileUtil.get_file_path("main/conf/config.yml"), 'r') as file:
            self.cfg = yaml.load(file, Loader=Loader)

    def get(self, key):
        keys = key.split(':')
        if len(keys) > 1:
            return self.__get_nest_value(self.cfg[keys[0]], keys[1:])
        else:
            return self.cfg[key]

    def set(self, key, value):
        self.cfg.__setitem__(key, value)

    def __get_nest_value(self, map_, keys):
        if len(keys) > 1:
            return self.__getNestedValue(map_[keys[0]], keys[1:])
        else:
            return map_[keys[0]]

    def dump(self):
        return yaml.dump(self.cfg, Dumper=Dumper)



