import torch.nn as nn

from main.common.common import *


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.lstm = nn.LSTMCell(conf.get('emb-size'), conf.get('dec-hidden-size'))

        self.y_concat = nn.Linear(2 * conf.get('enc-hidden-size') + conf.get('emb-size'), conf.get('emb-size'))

    '''
        :params
            y               : B, E
            pre_hidden      : B, DH
            pre_cell        : B, DH
            enc_ctx_vector  : B, 2EH
            
        :returns
            hidden          : B, DH
            cell            : B, DH   
    '''
    def forward(self, y, pre_hidden, pre_cell, enc_ctx_vector):
        y = self.y_concat(t.cat([y, enc_ctx_vector], dim=1))

        hidden, cell = self.lstm(y, (pre_hidden, pre_cell))

        return hidden, cell
