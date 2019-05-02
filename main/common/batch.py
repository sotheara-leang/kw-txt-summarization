from main.common.vocab import *
from main.common.common import *


class Batch(object):

    def __init__(self, articles,
                 articles_len,
                 articles_padding_mask,
                 extend_vocab_articles,
                 oovs,
                 keywords,
                 keywords_len,
                 summaries,
                 summaries_len,
                 original_summaries):

        self.articles = articles
        self.articles_len = articles_len
        self.extend_vocab_articles = extend_vocab_articles
        self.articles_padding_mask = articles_padding_mask

        self.summaries = summaries
        self.summaries_len = summaries_len

        self.oovs = oovs

        self.keywords = keywords
        self.keywords_len = keywords_len

        # to evaluate rouge score
        self.original_summaries = original_summaries


class BatchInitializer(object):

    def __init__(self, vocab, max_enc_steps, max_dec_steps, pointer_generator):
        self.vocab = vocab
        self.max_enc_steps = max_enc_steps
        self.max_dec_steps = max_dec_steps
        self.pointer_generator = pointer_generator

    def init(self, samples):
        articles, summaries, keywords = list(zip(*samples))

        # article
        articles_words = []
        for article in articles:
            words = self.truncate(article.split(), self.max_enc_steps)
            articles_words.append(words)

        articles_len = [len(a) + 1 for a in articles_words]       # +1 for STOP
        max_article_len = max(articles_len)

        enc_articles = []
        enc_extend_vocab_articles = []
        enc_articles_padding_mask = []
        oovs = []

        for article_words in articles_words:
            enc_article = self.vocab.words2ids(article_words) + [TK_STOP['id']]

            enc_article_padding_mask = [0] * len(enc_article) + [1] * (max_article_len - len(enc_article))

            enc_article += [TK_PADDING['id']] * (max_article_len - len(enc_article))

            if self.pointer_generator is True:
                enc_extend_vocab_article, article_oovs = self.vocab.extend_words2ids(article_words)

                enc_extend_vocab_article += [TK_STOP['id']]
                enc_extend_vocab_article += [TK_PADDING['id']] * (max_article_len - len(enc_extend_vocab_article))
            else:
                enc_extend_vocab_article = enc_article
                article_oovs = []

            enc_articles.append(enc_article)
            enc_articles_padding_mask.append(enc_article_padding_mask)
            enc_extend_vocab_articles.append(enc_extend_vocab_article)
            oovs.append(article_oovs)

        # summary
        summaries_words = []
        for summary in summaries:
            words = self.truncate(summary.split(), self.max_dec_steps)
            summaries_words.append(words)

        summaries_len = [len(s) + 2 for s in summaries_words]  # +2 for START & STOP
        max_summary_len = max(summaries_len)

        enc_summaries = []
        for i, summary_words in enumerate(summaries_words):
            enc_summary = [TK_START['id']] + self.vocab.words2ids(summary_words, oovs[i]) + [TK_STOP['id']]
            enc_summary += [TK_PADDING['id']] * (max_summary_len - len(enc_summary))

            enc_summaries.append(enc_summary)

        # keywords
        keywords_len = [len(kws.split()) + 1 for kws in keywords]

        max_keyword_len = max(keywords_len)

        enc_keywords = []
        for kws in keywords:
            ks_words = kws.split()

            enc_keyword = self.vocab.words2ids(ks_words)
            enc_keyword += [TK_PADDING['id']] * (max_keyword_len - len(enc_keyword))

            enc_keywords.append(enc_keywords)

        # covert to tensor
        enc_articles = cuda(t.tensor(enc_articles))
        articles_len = cuda(t.tensor(articles_len))

        enc_extend_vocab_articles = cuda(t.tensor(enc_extend_vocab_articles))
        enc_articles_padding_mask = cuda(t.ByteTensor(enc_articles_padding_mask))

        enc_summaries = cuda(t.tensor(enc_summaries))
        summaries_len = cuda(t.tensor(summaries_len))

        enc_keywords = cuda(t.tensor(enc_keywords))
        keywords_len = cuda(t.tensordot(keywords_len))

        # sort tensor
        articles_len, indices = articles_len.sort(0, descending=True)
        enc_articles = enc_articles[indices]

        return Batch(enc_articles,
                     articles_len,
                     enc_articles_padding_mask,
                     enc_extend_vocab_articles,
                     oovs,
                     enc_keywords,
                     keywords_len,
                     enc_summaries,
                     summaries_len,
                     summaries)

    def truncate(self, words, length):
        if len(words) > length:
            return words[:length]
        return words