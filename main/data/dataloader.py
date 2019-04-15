from main.common.dataloader import *


class DataLoader(DataLoader):

    SEPARATOR = '$$$'

    def __init__(self, article_file, summary_file, keyword_file, batch_size):
        super(DataLoader, self).__init__(batch_size)

        self.article_file = article_file
        self.summary_file = summary_file
        self.keyword_file = keyword_file

    def reader(self):
        with open(self.article_file, 'r') as art_reader, \
                open(self.summary_file, 'r') as sum_reader, open(self.summary_file, 'r') as kw_reader:
            while True:
                article = next(art_reader).strip()

                summaries = next(sum_reader).split(DataLoader.SEPARATOR)
                summaries = [summary.strip() for summary in summaries]

                kws = next(kw_reader).split(DataLoader.SEPARATOR)
                kws = [kw.strip() for kw in kws]

                yield article, summaries, kws



