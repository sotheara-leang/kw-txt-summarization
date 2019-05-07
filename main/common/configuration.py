import logging
import logging.config
import re
import os
import yaml
from yaml import Loader, Dumper

from main.common.util.file_util import FileUtil
from main.common.util.dict_util import DictUtil


class Configuration:

    def __init__(self, conf_file, log_file):
        with open(FileUtil.get_file_path(conf_file), 'r') as file:
            self.cfg = yaml.load(file, Loader=Loader)

        self.init_logging(log_file)

    def init_logging(self, log_file):
        with open(FileUtil.get_file_path(log_file), 'r') as f:
            param_matcher = re.compile(r'.*\$\{([^}^{]+)\}.*')

            def param_constructor(loader, node):
                value = node.value

                params = param_matcher.findall(value)
                for param in params:

                    param_value = self.get(param)
                    if param_value is None:
                        try:
                            param_value = os.environ[param]
                        except Exception:
                            pass

                    if param_value is not None:
                        value = value.replace('${' + param + '}', param_value)

                return value

            class VariableLoader(yaml.SafeLoader):
                pass

            VariableLoader.add_implicit_resolver('!param', param_matcher, None)
            VariableLoader.add_constructor('!param', param_constructor)

            config = yaml.load(f.read(), Loader=VariableLoader)

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
