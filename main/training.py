
from main.common.vocab import *
from main.common.common import *
from main.common.dataloader import *
from main.seq2seq import Seq2Seq


class Training(object):

    def __init__(self):
        self.training_args = conf.get('training')

        self.vocab = Vocab(self.training_args['vocab-file'])

        self.seq2seq = Seq2Seq(self.vocab)

        self.dataloader = DataLoader(FileUtil.get_file_path(self.training_args['article-file']),
                                     FileUtil.get_file_path(self.training_args['summary-file']),
                                     self.training_args['batch-size'])

        self.epoch = self.training_args['epoch']

        self.batch_initializer = BatchInitializer(self.vocab)

    def train_batch(self, batch):

        pass

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

                enc_articles, enc_summaries, art_extend_vocab, art_oovs = self.batch_initializer.init(batch)


                self.train_batch(batch)

                batch_counter += 1

            self.dataloader.reset()


if __name__ == "__main__":
    training = Training()
    training.run()
