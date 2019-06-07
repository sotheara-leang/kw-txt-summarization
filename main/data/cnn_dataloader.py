from main.common.dataloader import *


class CNNDataLoader(DataLoader):

    SEP_KEYWORD = '#E#'
    SEP_SUMMARY = '#S#'
    SEP_SUMMARY_QUERY = '#Q#'

    def __init__(self, article_file, summary_file, keyword_file, batch_size):
        self.logger = logger(self)

        super(CNNDataLoader, self).__init__(batch_size)

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

                    except StopIteration as e:
                        yield None
                    except Exception as e:
                        self.logger.error(e, exc_info=True)
                        raise e
        except IOError as e:
            self.logger.error(e, exc_info=True)
            raise e
