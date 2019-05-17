from unittest import TestCase

from main.common.util.file_util import FileUtil
from main.data.cnn_dataloader import CNNDataLoader
from main.common.batch import *
from main.common.simple_vocab import *

from test.common import *


class TestCNNDataLoader(TestCase):

    def test(self):
        dataloader = CNNDataLoader(FileUtil.get_file_path(conf('train:article-file')),
                                     FileUtil.get_file_path(conf('train:summary-file')),
                                     FileUtil.get_file_path(conf('train:keyword-file')), 2)

        max_enc_steps = conf('max-enc-steps')
        max_dec_steps = conf('max-dec-steps')

        vocab = SimpleVocab(FileUtil.get_file_path(conf('vocab-file')), conf('vocab-size'))

        batch_initializer = BatchInitializer(vocab, max_enc_steps, max_dec_steps, True)

        while True:
            batch = dataloader.next_batch()
            if batch is None:
                break

            batch = batch_initializer.init(batch)

            print(batch)
