import time

from main.common.vocab import *
from main.common.common import *
from main.common.batcher import Batcher
from main.seq2seq import Seq2Seq


class Training(object):

    def __init__(self):

        training_args = conf.get('training')

        self.vocab = Vocab(training_args['vocab-file'])
        self.batcher = Batcher(training_args['article-file'], training_args['summary-file'], self.vocab, training_args['batch-size'])

        conf.set('vocab-size', self.vocab.size())

        self.seq2seq = Seq2Seq(self.vocab)

    def train_batch(self, batch):

        pass

    def train_ml(self, batch):
        pass

    def train_rl(self, batch):
        pass

    def run(self):
        while True:
            batch = self.batcher.next_batch()


if __name__ == "__main__":
    training = Training()
    training.run()
