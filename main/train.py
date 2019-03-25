import torch.nn as nn
from rouge import Rouge

from main.data.giga import *
from main.seq2seq import Seq2Seq
from main.common.batch import *


class Train(object):

    def __init__(self):
        self.epoch = conf.get('train:epoch')

        self.vocab = Vocab(conf.get('train:vocab-file'))

        self.seq2seq = Seq2Seq(self.vocab)

        self.batch_initializer = BatchInitializer(self.vocab, conf.get('max-enc-steps'))

        self.dataloader = GigaDataLoader(FileUtil.get_file_path(conf.get('train:article-file')), FileUtil.get_file_path(conf.get('train:summary-file')), conf.get('train:batch-size'))

        #
        self.optimizer = t.optim.Adagrad(self.seq2seq.parameters(), lr=conf.get('train:lr'))

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        #
        self.rouge = Rouge()

    def train_batch(self, batch):
        # ML
        output = self.seq2seq(
            batch.articles,
            batch.articles_len,
            batch.summaries, batch.extend_vocab, batch.max_ovv_len, criterion=self.criterion, teacher_forcing=True, greedy_search=False)

        # RL
        sample_output = self.seq2seq(
            batch.articles,
            batch.articles_len,
            batch.summaries, batch.extend_vocab, batch.max_ovv_len, criterion=self.criterion, greedy_search=False)

        baseline_output = self.seq2seq(
            batch.articles,
            batch.articles_len,
            batch.summaries, batch.extend_vocab, batch.max_ovv_len)

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
    train = Train()
    train.run()
