import logging
import logging.config
import yaml
from yaml import Loader, Dumper
from singleton_decorator import singleton
from main.common.util.file_util import FileUtil
from main.common.util.dict_util import DictUtil


@singleton
class Configuration:

    def __init__(self):
        with open(FileUtil.get_file_path("main/conf/config.yml"), 'r') as file:
            self.cfg = yaml.load(file, Loader=Loader)

        with open(FileUtil.get_file_path("main/conf/logging.yml"), 'r') as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)

    def exist(self, key):
        return True if self.get(key) is not None else False

    def get(self, key, default=None):
        value = None
        try:
            keys = key.split(':')
            if len(keys) > 1:
                value = self.__get_nest_value(self.cfg[keys[0]], keys[1:])
            else:
                value = self.cfg[key]

            if value is None and default is not None:
                value = default

        except Exception:
            pass

        return value

    def set(self, key, value):
        self.cfg.__setitem__(key, value)

    def __get_nest_value(self, map_, keys):
        if len(keys) > 1:
            return self.__get_nest_value(map_[keys[0]], keys[1:])
        else:
            return map_[keys[0]]

    def dump(self):
        return yaml.dump(self.cfg, Dumper=Dumper)

    def merge(self, conf_file):
        with open(FileUtil.get_file_path(conf_file), 'r') as file:
            cfg = yaml.load(file, Loader=Loader)

            DictUtil.dict_merge(self.cfg, cfg)
