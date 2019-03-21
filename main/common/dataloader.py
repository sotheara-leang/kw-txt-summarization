from main.common.common import *
from main.common.vocab import *


class BatchInitializer(object):

    def __init__(self, vocab):
        self.vocab = vocab

    def init(self, samples):
        max_enc_steps = conf.get('max-enc-steps')
        pad_token_id = self.vocab.word2id(PAD_TOKEN)

        articles, summaries = list(zip(*samples))

        max_article_len = max([len(a.split()) for a in articles])
        max_summary_len = max([len(s.split()) for s in summaries])

        enc_articles = []
        enc_summaries = []

        # article
        for article in articles:
            art_words = article.split()
            if len(art_words) > max_enc_steps:  # truncate
                art_words = art_words[:max_enc_steps]

            art_extend_vocab, art_oovs = article2ids(art_words, self.vocab)

            enc_article = [self.vocab.word2id(w) for w in art_words]
            while len(enc_article) < max_article_len:
                enc_article.append(pad_token_id)

            enc_articles.append(enc_article)

        # summary
        for summary in summaries:
            summary_words = summary.split()
            if len(summary_words) > max_enc_steps:  # truncate
                summary_words = summary_words[:max_enc_steps]

            enc_summary = summary2ids(summary_words, self.vocab, art_oovs) + [self.vocab.word2id(STOP_DECODING)]
            while len(enc_summary) < max_summary_len:
                enc_summary.append(pad_token_id)

            enc_summaries.append(enc_summary)

        return enc_articles, enc_summaries, art_extend_vocab, art_oovs


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


