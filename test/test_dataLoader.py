from unittest import TestCase

from main.common.util.file_util import FileUtil
from main.common.dataloader import *


class TestDataLoader(TestCase):

    def test(self):
        args = conf.get('training')
        dataloader = DataLoader(
            FileUtil.get_file_path(args['article-file']),
            FileUtil.get_file_path(args['summary-file']), 15)

        while True:
            batch = dataloader.next()
            if batch is None:
                break








