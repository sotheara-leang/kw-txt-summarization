import time

from main.common.vocab import *
from main.common.common import *
from main.common.batcher import Batcher


class Training(object):

    def __init__(self):

        args = conf.get('training')

        self.vocab = Vocab(args['vocab-file'])
        self.batcher = Batcher(args['article-file'], args['summary-file'], self.vocab, args['batch-size'])

        conf.set('vocab-size', self.vocab.size())

    def train_one_batch(self, batch):
        pass

    def run(self):

        while True:
            batch = self.batcher.next_batch()



if __name__ == "__main__":
    training = Training()
    training.run()
