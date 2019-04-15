import torch as t
import torch.nn as nn

from main.common.common import *


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.lstm = nn.LSTMCell(conf.get('emb-size'), 2 * conf.get('hidden-size'))

        self.y_concat = nn.Linear(2 * conf.get('hidden-size') + conf.get('emb-size'), conf.get('emb-size'))

    '''
        :params
            y               : B, E
            pre_hidden      : B, 2H
            pre_cell        : B, 2H
        :returns
            hidden          : B, 2H
            cell            : B, 2H   
    '''
    def forward(self, y, pre_hidden, pre_cell, enc_ctx_vector):
        y_concat = self.y_concat(t.cat([y, enc_ctx_vector], dim=1))

        hidden, cell = self.lstm(y_concat, (pre_hidden, pre_cell))
        return hidden, cell
