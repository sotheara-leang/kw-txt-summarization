import logging.config
import re
import collections
import os
import yaml

from main.common.common import *
from main.common.util.file_util import FileUtil


class Logger(object):

    def __init__(self, log_file):
        with open(FileUtil.get_file_path(log_file), 'r') as f:
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

            config = yaml.load(f.read(), Loader=VariableLoader)

            self.init_log_dir(config)

            logging.config.dictConfig(config)

    def init_log_dir(self, dict_):
        for k, v in dict_.items():
            if k == 'filename':
                filename = v
                file_dir = os.path.dirname(os.path.abspath(filename))

                if not os.path.exists(file_dir):
                    os.makedirs(file_dir)
            elif isinstance(v, collections.Mapping):
                self.init_log_dir(v)
