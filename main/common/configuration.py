import os
import re

import yaml
from yaml import Loader, Dumper

from main.common.util.dict_util import DictUtil
from main.common.util.file_util import FileUtil


class Configuration:

    def __init__(self, conf_file):
        with open(FileUtil.get_file_path(conf_file), 'r') as file:
            param_matcher = re.compile(r'.*\$\{([^}^{]+)\}.*')

            def param_constructor(loader, node):
                value = node.value

                params = param_matcher.findall(value)
                for param in params:
                    try:
                        param_value = os.environ[param]
                        return value.replace('${' + param + '}', param_value)
                    except Exception:
                        pass

                return value

            class VariableLoader(yaml.SafeLoader):
                pass

            VariableLoader.add_implicit_resolver('!param', param_matcher, None)
            VariableLoader.add_constructor('!param', param_constructor)

            self.cfg = yaml.load(file, Loader=VariableLoader)

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

            DictUtil.merge(self.cfg, cfg)
