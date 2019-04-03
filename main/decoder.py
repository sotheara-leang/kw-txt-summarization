import torch.nn as nn

from main.common.common import *


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        emb_size = conf.get('emb-size')
        hidden_size = conf.get('hidden-size')

        self.combine_x = nn.Linear(2 * hidden_size + emb_size, emb_size)

        self.lstm = nn.LSTMCell(emb_size, 2 * hidden_size)

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
        x = t.cat([y, enc_ctx_vector], dim=1)

        x = self.combine_x(x)

        hidden, cell = self.lstm(x, (pre_hidden, pre_cell))  # B, 2H

        return hidden, cell
