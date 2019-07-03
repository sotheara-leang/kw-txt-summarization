import pickle

import torch.nn as nn

from main.common.vocab import *


class GloveEmbedding(nn.Embedding):

    def __init__(self, emb_file, vocab, freeze=True):
        self.logger = logger(self)

        self.logger.debug('initialize embedding from: %s', emb_file)

        data = self.load_emb(emb_file)

        word2vect = data['word2vect']

        glove_embedding = np.asarray(list(word2vect.values()))

        mean = np.mean(glove_embedding, axis=0)
        std = glove_embedding.std(axis=0)

        vocab_size = vocab.size()
        emb_size = glove_embedding.shape[1]

        embedding = np.random.normal(mean, std, [vocab_size, emb_size])

        for id_, word in vocab.id2word_map().items():
            if word in word2vect:
                embedding[id_] = word2vect[word]

        super(GloveEmbedding, self).__init__(num_embeddings=vocab_size, embedding_dim=emb_size, padding_idx=TK_PADDING['id'], _weight=t.FloatTensor(embedding))

        self.weight.requires_grad = not freeze

    def load_emb(self, emb_file):
        with open(emb_file, 'rb') as f:
            return pickle.load(f)
