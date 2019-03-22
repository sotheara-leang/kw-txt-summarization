from main.common.vocab import *


class Batch(object):

    def __init__(self, enc_articles, enc_summaries, oovs):
        self.enc_articles = enc_articles
        self.enc_summaries = enc_summaries
        self.oovs = oovs

        self.max_ovv_len = [len(ovv) for ovv in oovs]


class BatchInitializer(object):

    def __init__(self, vocab):
        self.vocab = vocab

    def init(self, samples):
        max_enc_steps = conf.get('max-enc-steps')
        pad_token_id = self.vocab.word2id(PAD_TOKEN)

        articles, summaries = list(zip(*samples))

        article_len = [len(a.split()) for a in articles]
        summary_len = [len(s.split()) for s in summaries]

        max_article_len = max(article_len)
        max_summary_len = max(summary_len)

        enc_articles = []
        enc_summaries = []
        oovs = []

        # article
        for article in articles:
            art_words = article.split()
            if len(art_words) > max_enc_steps:  # truncate
                art_words = art_words[:max_enc_steps]

            enc_article, article_oovs = article2ids(art_words, self.vocab)

            while len(enc_article) < max_article_len:
                enc_article.append(pad_token_id)

            enc_articles.append(enc_article)
            oovs.append(article_oovs)

        # summary
        for summary in summaries:
            summary_words = summary.split()
            if len(summary_words) > max_enc_steps:  # truncate
                summary_words = summary_words[:max_enc_steps]

            enc_summary = summary2ids(summary_words, self.vocab, oovs) + [self.vocab.word2id(STOP_DECODING)]
            while len(enc_summary) < max_summary_len:
                enc_summary.append(pad_token_id)

            enc_summaries.append(enc_summary)

        #

        return Batch(enc_articles, enc_summaries, oovs)


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
                return None

            if sample is None:
                break
            samples.append(sample)

        return samples

    def reset(self):
        self.generator = self.sample_generator()


