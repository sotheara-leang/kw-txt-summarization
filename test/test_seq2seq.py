from unittest import TestCase

from main.common.common import *
from main.seq2seq import Seq2Seq
from main.common.vocab import Vocab
from main.common.util.file_util import FileUtil


class TestConfiguration(TestCase):

    def test(self):
        vocab = Vocab(FileUtil.get_file_path(conf.get('vocab-file')))

        model = Seq2Seq(vocab)
        model.load_state_dict(t.load(FileUtil.get_file_path(conf.get('model-file')), map_location='cpu'))

        model.eval()

        article = 'south korea on monday announced sweeping tax reforms , including income and corporate tax cuts to boost growth by stimulating sluggish private consumption and business investment .'

        summary = model.summarize(article)

        print(summary)





