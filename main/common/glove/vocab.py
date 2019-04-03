import pickle

from main.common.vocab import *


class GloveVocab(Vocab):

    def __init__(self, vocab_file):
        super(GloveVocab, self).__init__({}, {}, {})

        with open(vocab_file, 'rb') as vocab_f:
            vocab_ = pickle.load(vocab_f)

            self._word2id = vocab_['word2id']
            self._id2word = vocab_['id2word']

            for token in [TK_PADDING, TK_UNKNOWN, TK_START_DECODING, TK_STOP_DECODING]:
                self._word2id[token['word']] = token['id']
                self._id2word[token['id']] = token['word']
