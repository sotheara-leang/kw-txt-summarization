from unittest import TestCase

from main.common.glove.vocab import *
from main.common.util.file_util import FileUtil
from main.common.glove.embedding import GloveEmbedding


class TestConfiguration(TestCase):

    def test(self):
        vocab = GloveVocab(FileUtil.get_file_path('data/extract/vocab.bin'),
                           FileUtil.get_file_path('data/extract/embedding'))

        emb = GloveEmbedding(vocab)




