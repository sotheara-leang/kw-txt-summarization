import torch as t
import torch.nn as nn

from main.common.common import *


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(conf.get('emb-size'), conf.get('hidden-size'), num_layers=1, batch_first=True,
                            bidirectional=True)

    '''
        :param
            x       : B, L, H
            seq_len : B, L 
            
        :return
            outputs     : B, L, 2H
            hidden_n    : B, 2H
            cell_n      : B, 2H
    '''
    def forward(self, x, seq_len):
        packed_x = nn.pack_padded_sequence(x, seq_len, batch_first=True)

        # output    : B, L, 2H
        # hidden_n  : 2, B, H
        # cell_n    : 2, B, H
        outputs, hidden_n, cell_n = self.lstm(packed_x)

        outputs, _ = nn.pad_packed_sequence(outputs, batch_first=True).contiguous()

        # B, 2H
        hidden_n = t.cat(list(hidden_n), dim=1)

        # B, 2H
        cell_n = t.cat(list(cell_n), dim=1)

        return outputs, (hidden_n, cell_n)



