import torch.nn as nn

from main.encoder import Encoder
from main.decoder import Decoder

from main.common.common import *


class Model(nn.Module):

    def __int__(self):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(conf.get('vocab-size'), conf.get('emb-size'))
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, flag='decode'):
        if flag == 'train-ml':
            pass
        elif flag == 'train-rl':
            pass
        else:
            pass



