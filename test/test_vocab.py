from unittest import TestCase

from main.common.util.file_util import FileUtil
from main.common.common import *
from main.common.simple_vocab import SimpleVocab


class TestVocab(TestCase):

    def test(self):
        vocab = SimpleVocab(FileUtil.get_file_path(conf.get('train:vocab-file')), conf.get('vocab-size'))

        print(vocab.word2id('australia'))

        words = 'doctors madrid tone thyda'.split(' ')

        ids = vocab.words2ids(words)

        print(ids)

        ids, oov = vocab.extend_words2ids(words)

        print(ids, oov)

        n_words = vocab.ids2words(ids)

        print(n_words)

        n_words = vocab.ids2words(ids, oov)

        print(n_words)



