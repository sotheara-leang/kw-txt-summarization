from unittest import TestCase

from rouge import Rouge
from random import randint

from main.seq2seq import Seq2Seq
from main.common.simple_vocab import *
from main.common.simple_vocab import SimpleVocab
from main.data.cnn_dataloader import *
from test.common import *


class TestSeq2Seq(TestCase):

    def get_score(self, summary, reference):
        rouge = Rouge()

        summary = summary.split()
        summary = [w for w in summary if w != TK_STOP['word']]

        score = rouge.get_scores(' '.join(summary), reference)[0]["rouge-l"]["f"]

        return score

    def test(self):
        data_loader = CNNDataLoader(FileUtil.get_file_path(conf('train:article-file')),
                                    FileUtil.get_file_path(conf('train:summary-file')),
                                    FileUtil.get_file_path(conf('train:keyword-file')),
                                    conf('train:batch-size'))

        vocab = SimpleVocab(FileUtil.get_file_path(conf('vocab-file')), conf('vocab-size'))

        seq2seq = cuda(Seq2Seq(vocab))

        checkpoint = t.load(FileUtil.get_file_path(conf('model-file')))

        seq2seq.load_state_dict(checkpoint['model_state_dict'])

        seq2seq.eval()

        samples = data_loader.read_all()

        article, keyword, reference = samples[randint(0, len(samples) - 1)]

        keyword = keyword[0]
        reference = reference[0]

        summary, attention = seq2seq.evaluate(article, keyword)

        score = self.get_score(summary, reference)

        print('>>> article: ', article)
        print('>>> keyword: ', keyword)
        print('>>> reference: ', reference)
        print('========================')
        print('>>> prediction: ', summary)
        print('>>> score: ', score)






