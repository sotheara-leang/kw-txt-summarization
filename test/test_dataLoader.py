from unittest import TestCase

from main.common.dataloader import *


class TestDataLoader(TestCase):

    def test(self):
        dataloader = DataLoader(
            FileUtil.get_file_path(conf.get('train:article-file')),
            FileUtil.get_file_path(conf.get('train:summary-file')), 15)

        batch = dataloader.next()

        print(batch)








