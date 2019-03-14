import torch.nn as nn

from main.common.common import *


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(conf.get('emb-size'), conf.get('hidden-size'), num_layers=1, batch_first=True,
                            bidirectional=True)

    def forward(self, x, seq_lens):
        pass
