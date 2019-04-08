import torch as t
import torch.nn as nn
import bcolz
import numpy as np
from main.common.vocab import *


class GloveEmbedding(nn.Module):

    def __init__(self, emb_file):
        super(GloveEmbedding, self).__init__()

        vectors = bcolz.open(emb_file)[:]

        default_vectors = []
        for token in [TK_PADDING, TK_UNKNOWN, TK_START, TK_STOP]:
            vector = np.zeros(vectors.shape[1])
            vector[token['id']] = 1 if token['id'] != 0 else 0

            default_vectors.append(vector)

        default_vectors = np.asarray(default_vectors)

        vectors = np.concatenate((default_vectors, vectors), axis=0)

        n_vocab, vocab_dim = vectors.shape

        self.embedding = nn.Embedding(n_vocab, vocab_dim, padding_idx=0)
        self.embedding.from_pretrained(t.tensor(vectors))

    def forward(self, x):
        return self.embedding(x)






