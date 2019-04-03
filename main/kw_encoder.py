import torch as t
import torch.nn as nn


class KWEncoder(nn.Module):

    def __init__(self, embedding):
        super(KWEncoder, self).__init__()

        self.embedding = embedding

    '''
        :param
            kw     : B, L
        :return
            emb    : B
    '''
    def forward(self, kw):
        emb = self.embedding(kw)
        emb = t.sum(emb, dim=1)
        return emb
