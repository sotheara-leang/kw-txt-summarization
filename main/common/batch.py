from main.common.vocab import *
from main.common.common import *


class Batch(object):

    def __init__(self, articles, articles_len, articles_padding_mask, extend_vocab_articles, oovs, summaries, summaries_len, original_summaries):
        self.articles = articles
        self.articles_len = articles_len
        self.extend_vocab_articles = extend_vocab_articles
        self.articles_padding_mask = articles_padding_mask

        self.summaries = summaries
        self.summaries_len = summaries_len

        self.oovs = oovs

        # to evaluate rouge score
        self.original_summaries = original_summaries


class BatchInitializer(object):

    def __init__(self, vocab, max_enc_steps):
        self.vocab = vocab
        self.max_enc_steps = max_enc_steps

    def init(self, samples):
        articles, summaries = list(zip(*samples))

        articles_len = [len(a.split()) + 1 for a in articles]       # +1 for STOP_DECODING
        summaries_len = [len(s.split()) + 1 for s in summaries]     # +1 for STOP_DECODING

        max_article_len = max(articles_len)
        max_summary_len = max(summaries_len)

        enc_articles = []
        enc_extend_vocab_articles = []
        enc_articles_padding_mask = []
        oovs = []

        # article
        for article in articles:
            art_words = article.split()
            if len(art_words) > self.max_enc_steps:  # truncate
                art_words = art_words[:self.max_enc_steps]

            enc_article = self.vocab.words2ids(art_words) + [TK_STOP['id']]
            enc_article += [TK_PADDING['id']] * (max_article_len - len(enc_article))

            enc_extend_vocab_article, article_oovs = self.vocab.extend_words2ids(art_words)
            enc_extend_vocab_article += [TK_PADDING['id']] * (max_article_len - len(enc_extend_vocab_article))

            enc_article_padding_mask = [1] * (len(art_words) + 1) + [0] * (max_article_len - len(art_words) - 1)

            enc_articles.append(enc_article)
            enc_extend_vocab_articles.append(enc_extend_vocab_article)
            enc_articles_padding_mask.append(enc_article_padding_mask)
            oovs.append(article_oovs)

        # summary
        enc_summaries = []

        for summary in summaries:
            summary_words = summary.split()
            if len(summary_words) > self.max_enc_steps:  # truncate
                summary_words = summary_words[:self.max_enc_steps]

            enc_summary = self.vocab.words2ids(summary_words, oovs)
            enc_summary = enc_summary + [TK_STOP['id']]
            enc_summary += [TK_PADDING['id']] * (max_summary_len - len(enc_summary))

            enc_summaries.append(enc_summary)

        # covert to tensor
        enc_articles = cuda(t.tensor(enc_articles))
        articles_len = cuda(t.tensor(articles_len))

        enc_extend_vocab_articles = cuda(t.tensor(enc_extend_vocab_articles))
        enc_articles_padding_mask = cuda(t.tensor(enc_articles_padding_mask))

        enc_summaries = cuda(t.tensor(enc_summaries))
        summaries_len = cuda(t.tensor(summaries_len))

        # sort tensor
        articles_len, indices = articles_len.sort(0, descending=True)
        enc_articles = enc_articles[indices]

        return Batch(enc_articles,
                     articles_len,
                     enc_articles_padding_mask,
                     enc_extend_vocab_articles,
                     oovs,
                     enc_summaries,
                     summaries_len,
                     summaries)


