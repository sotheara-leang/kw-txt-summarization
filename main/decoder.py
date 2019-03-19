import torch as t
import torch.nn as nn

from main.common.common import *


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        emb_size = conf.get('emb-size')
        hidden_size = conf.get('hidden-size')

        self.x_concat = nn.Linear(2 * hidden_size + emb_size, emb_size)

        self.lstm = nn.LSTMCell(emb_size, 2 * hidden_size)

    '''
        :param
            x               : B, 2H
            pre_hidden      : B, 2H
            ctx_vector      : B, 2H
        
        :return
           
    '''
    def forward(self, x, pre_hidden, ctx_vector):
        # concatenate input with encoder context vector
        x = self.x_concat(t.cat([x, ctx_vector], dim=1))    # B, E + 2H

        # new decoder state
        hidden, _ = self.lstm(x, pre_hidden)  # B, 2H

        return hidden
