from unittest import TestCase

from main.common.common import *
from main.seq2seq import Seq2Seq
from main.common.simple_vocab import SimpleVocab
from main.common.util.file_util import FileUtil
from main.data.giga import GigaDataLoader


class TestConfiguration(TestCase):

    def test(self):
        data_loader = GigaDataLoader(FileUtil.get_file_path(conf.get('train:article-file')),
                                         FileUtil.get_file_path(conf.get('train:summary-file')), 2)

        vocab = SimpleVocab(FileUtil.get_file_path(conf.get('vocab-file')), conf.get('vocab-size'))

        model = Seq2Seq(vocab)
        model.load_state_dict(t.load(FileUtil.get_file_path(conf.get('model-file')), map_location='cpu'))

        model.eval()

        article, _ = data_loader.next()

        summary = model.summarize(article)

        print(summary)





