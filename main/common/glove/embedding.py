import torch as t
import torch.nn as nn

from main.common.glove.vocab import GloveVocab


class GloveEmbedding(nn.Module):

    def __init__(self, vocab: GloveVocab):
        super(GloveEmbedding, self).__init__()

        n_vocab, vocab_dim = vocab.vectors.shape

        self.embedding = nn.Embedding(n_vocab, vocab_dim, padding_idx=0)
        self.embedding.from_pretrained(t.tensor(vocab.vectors))

    def forward(self, x):
        return self.embedding(x)






