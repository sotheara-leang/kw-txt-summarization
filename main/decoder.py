import torch.nn as nn

from main.common.common import *


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.lstm = nn.LSTMCell(conf.get('emb-size'), 2 * conf.get('hidden-size'))

    '''
        :params
            y               : B, E
            pre_hidden      : B, 2H
            pre_cell        : B, 2H
        :returns
            hidden          : B, 2H
            cell            : B, 2H   
    '''
    def forward(self, y, pre_hidden, pre_cell):
        hidden, cell = self.lstm(y, (pre_hidden, pre_cell))
        return hidden, cell
