from main.common.dataloader import *


class GigaDataLoader(DataLoader):

    def __init__(self, article_file, summary_file, batch_size):
        super(GigaDataLoader, self).__init__(batch_size)

        self.article_file = article_file
        self.summary_file = summary_file

    def reader(self):
        with open(self.article_file, 'r') as art_reader, open(self.summary_file, 'r') as sum_reader:
            while True:
                article = next(art_reader)
                summary = next(sum_reader)

                yield article, summary



