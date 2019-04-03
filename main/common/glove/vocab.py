import pickle
import bcolz
import numpy as np

from main.common.vocab import *


class GloveVocab(Vocab):

    def __init__(self, vocab_file, embedding_file):
        super(GloveVocab, self).__init__({}, {}, {})

        with open(vocab_file, 'rb') as vocab_f:
            vocab_ = pickle.load(vocab_f)

            self._word2id = vocab_['word2id']
            self._id2word = vocab_['id2word']

            self.vectors = bcolz.open(embedding_file)[:]

            default_vectors = []
            for token in [TK_PADDING, TK_UNKNOWN, TK_START_DECODING, TK_STOP_DECODING]:
                self._word2id[token['word']] = token['id']
                self._id2word[token['id']] = token['word']

                vector = np.zeros(self.vectors.shape[1])
                vector[token['id']] = 1 if token['id'] != 0 else 0

                default_vectors.append(vector)

            default_vectors = np.asarray(default_vectors)

            self.vectors = np.concatenate((default_vectors, bcolz.open(embedding_file)[:]), axis=0)

            for id_ in self._id2word:
                self._id2vector[id_] = self.vectors[id_]


