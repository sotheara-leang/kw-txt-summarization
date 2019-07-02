from main.common.dataloader import *


class CNNDataLoader(DataLoader):

    SEP_SUMMARY = '#S#'

    def __init__(self, article_file, summary_file, keyword_file, batch_size, mode='train'):
        self.logger = logger(self)

        super(CNNDataLoader, self).__init__(batch_size, mode)

        self.article_file = article_file
        self.summary_file = summary_file
        self.keyword_file = keyword_file

    def reader(self):
        try:
            with open(self.article_file, 'r', encoding='utf-8') as art_reader, \
                    open(self.summary_file, 'r', encoding='utf-8') as sum_reader, \
                    open(self.keyword_file, 'r', encoding='utf-8') as kw_reader:
                while True:
                    try:
                        article = next(art_reader).strip()
                        summaries = next(sum_reader).strip()
                        kws = next(kw_reader).strip()

                        if article == '' or summaries == '':
                            continue

                        summaries = summaries.replace(CNNDataLoader.SEP_SUMMARY, ' ')

                        yield article, kws, summaries

                    except StopIteration:
                        yield None
        except IOError as e:
            self.logger.error(e, exc_info=True)
            raise e
