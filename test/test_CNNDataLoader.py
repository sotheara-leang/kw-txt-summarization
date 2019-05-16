from unittest import TestCase

from main.common.util.file_util import FileUtil
from main.data.cnn_dataloader import CNNDataLoader

from test.common import *


class TestCNNDataLoader(TestCase):

    def test(self):
        dataloader = CNNDataLoader(FileUtil.get_file_path(conf('train:article-file')),
                                     FileUtil.get_file_path(conf('train:summary-file')),
                                     FileUtil.get_file_path(conf('train:keyword-file')), 2)

        counter = 0
        while True:
            batch = dataloader.next_batch()
            if batch is None:
                break

            counter += len(batch)

            print(counter)

            print(batch)
