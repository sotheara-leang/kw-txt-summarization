from unittest import TestCase
from main.common.glove.vocab import *
from main.common.util.file_util import FileUtil


class TestConfiguration(TestCase):

    def test(self):
        GloveVocab(FileUtil.get_file_path('data/extract/vocab.bin'))




