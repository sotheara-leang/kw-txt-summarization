import torch.nn as nn
import torch.nn.functional as f

from main.common.common import *


class ReduceEncoder(nn.Module):

    def __init__(self):
        super(ReduceEncoder, self).__init__()

        self.reduce_h = nn.Linear(2 * conf.get('enc-hidden-size'), conf.get('dec-hidden-size'))

        self.reduce_c = nn.Linear(2 * conf.get('enc-hidden-size'), conf.get('dec-hidden-size'))

    '''
        :params
           hidden   :   2, B, EH 
           cell     :   2, B, EH
            
        :returns
           hidden   :   B, DH
           cell     :   B, DH
    '''
    def forward(self, hidden, cell):
        hidden = t.cat(list(hidden), dim=1)
        hidden = f.relu(self.reduce_h(hidden))

        cell = t.cat(list(cell), dim=1)
        cell = f.relu(self.reduce_c(cell))

        return hidden, cell
