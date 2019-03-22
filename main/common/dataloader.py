from main.common.vocab import *


class DataLoader(object):

    def __init__(self, article_file, summary_file, batch_size):
        self.article_file = article_file
        self.summary_file = summary_file
        self.batch_size = batch_size

        self.generator = self.sample_generator()

    def sample_generator(self):
        with open(self.article_file, 'r') as art_reader, open(self.summary_file, 'r') as sum_reader:
            while True:
                article = next(art_reader)
                summary = next(sum_reader)

                yield (article, summary)

    def next(self):
        samples = []
        for i in range(0, self.batch_size):
            try:
                sample = next(self.generator)
            except StopIteration:
                logger.debug('no more sample to read')
                return None

            if sample is None:
                break
            samples.append(sample)

        return samples

    def reset(self):
        logger.debug('reset data loader')

        self.generator = self.sample_generator()


