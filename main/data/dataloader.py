from main.common.dataloader import *


class DataLoader(DataLoader):

    def __init__(self, article_file, summary_file, keyword_file, batch_size):
        super(DataLoader, self).__init__(batch_size)

        self.article_file = article_file
        self.summary_file = summary_file
        self.keyword_file = keyword_file

    def reader(self):
        with open(self.article_file, 'r') as art_reader, \
                open(self.summary_file, 'r') as sum_reader, open(self.summary_file, 'r') as kw_reader:
            while True:
                article = next(art_reader)
                summary = next(sum_reader)
                kw = next(kw_reader)

                yield article.strip(), summary.strip(), kw



