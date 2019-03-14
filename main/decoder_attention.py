import torch as t
import torch.nn as nn
import torch.nn.functional as f

from main.common.common import *


class DecoderAttention(nn.Module):

    def __init__(self):
        super(DecoderAttention, self).__init__()

        self.attn = nn.Bilinear(2 * conf.get('hidden-size'), 2 * conf.get('hidden-size'), 1, False)

    '''
        dec_hidden  : 1, H
        pre_dec_hidden  : B, H
    '''
    def forward(self, dec_hidden, pre_dec_hidden):
        if pre_dec_hidden is None:
            context_vector = t.zeros(dec_hidden.size())
        else:
            score = self.attn(dec_hidden.t(), pre_dec_hidden)   # B, 1

            # softmax

            attention = f.softmax(score, dim=1)

            # context vector

            context_vector = t.bmm(attention.unsqueeze(1), pre_dec_hidden)  # B, 1, L * B, L, H  ->  B, 1, 2*H
            context_vector = context_vector.squeeze(1)  # B, 2*H

        return context_vector
