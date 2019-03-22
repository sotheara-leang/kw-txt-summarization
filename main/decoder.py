import torch as t
import torch.nn as nn

from main.common.common import *


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.gru = nn.GRUCell(conf.get('emb-size'), 2 * conf.get('hidden-size'))

    '''
        :param
            y               : B, E
            pre_hidden      : B, 2H
        
        :return
            hidden          : B, 2H
           
    '''
    def forward(self, y, pre_hidden):
        hidden = self.gru(y, pre_hidden)  # B, 2H
        return hidden
