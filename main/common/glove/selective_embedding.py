import torch.nn as nn
import numpy as np

from main.common.vocab import *


class GloveSelectiveEmbedding(nn.Embedding):

    def __init__(self, emb_file, vocab):
        word2id, id2word, id2vect = self.load_emb_file(emb_file, vocab)

        vectors = np.asarray(list(id2vect.values()))

        default_vectors = []
        for token in [TK_PADDING, TK_UNKNOWN, TK_START, TK_STOP]:
            vector = np.zeros(vectors.shape[1])
            vector[token['id']] = 1 if token['id'] != 0 else 0

            default_vectors.append(vector)

        default_vectors = np.asarray(default_vectors)

        vectors = np.concatenate((default_vectors, vectors), axis=0)
        n_vocab, vocab_dim = vectors.shape

        super(GloveSelectiveEmbedding, self).__init__(num_embeddings=n_vocab,
                                                      embedding_dim=vocab_dim, padding_idx=0, _weight=t.FloatTensor(vectors))

    def load_emb_file(self, emb_file, vocab):
        word2id = {}
        id2word = {}
        id2vect = {}

        id = 4

        with open(emb_file, 'r') as f:
            for line in f:
                line = line.split()

                word = line[0]

                if vocab.word2id(word) is None:
                    continue

                vector = np.array(line[1:]).astype(np.float)

                word2id[word] = id
                id2word[id] = word
                id2vect[id] = vector

                id += 1

        return word2id, id2word, id2vect







