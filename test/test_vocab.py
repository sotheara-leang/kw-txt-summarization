from unittest import TestCase

from main.common.vocab import *


class TestVocab(TestCase):

    def test(self):
        vocab = Vocab(FileUtil.get_file_path(conf.get('train:vocab-file')))
        vocab.write_metadata(FileUtil.get_file_path('data/vocab.txt'))
