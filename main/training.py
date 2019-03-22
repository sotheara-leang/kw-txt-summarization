import torch as t
import torch.nn as nn
from main.common.dataloader import *
from main.seq2seq import Seq2Seq
from main.common.batch import *


class Training(object):

    def __init__(self):
        self.training_args = conf.get('training')
        self.epoch = self.training_args['epoch']

        self.vocab = Vocab(self.training_args['vocab-file'])

        self.seq2seq = Seq2Seq(self.vocab)

        self.batch_initializer = BatchInitializer(self.vocab, conf.get('max-enc-steps'))
        self.dataloader = DataLoader(FileUtil.get_file_path(self.training_args['article-file']),
                                     FileUtil.get_file_path(self.training_args['summary-file']),
                                     self.training_args['batch-size'])

        #
        self.optimizer = t.optim.Adagrad(self.seq2seq.parameters(), lr=self.training_args['lr'])
        self.criterion = nn.NLLLoss(ignore_index=self.vocab.word2id(PAD_TOKEN))

    def train_batch(self, batch):
        y, y_prob = self.seq2seq(batch.articles, batch.articles_len, batch.summaries, batch.oovs, batch.oov_extra_zero, True)

    def train_ml(self, batch):
        pass

    def train_rl(self, batch):
        pass

    def run(self):
        for i in range(self.epoch):
            logger.debug('>>> Epoch %i/%i <<<', i+1, self.epoch)

            batch_counter = 1

            while True:
                logger.debug('Batch %i', batch_counter)

                batch = self.dataloader.next()

                if batch is None:
                    break

                batch = self.batch_initializer.init(batch)

                self.train_batch(batch)

                return

                batch_counter += 1

            self.dataloader.reset()


if __name__ == "__main__":
    training = Training()
    training.run()
