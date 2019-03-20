import torch as t
import torch.nn as nn
import torch.nn.functional as f

from main.common.common import *


class DecoderAttention(nn.Module):

    def __init__(self):
        super(DecoderAttention, self).__init__()

        self.attn = nn.Bilinear(2 * conf.get('hidden-size'), 2 * conf.get('hidden-size'), 1, False)

    '''
        :param
            dec_hidden       : B, 2H
            pre_dec_hiddens  : B, T, 2H
            
        :return
            context_vector   : B, 2*H
    '''
    def forward(self, dec_hidden, pre_dec_hiddens):
        if pre_dec_hiddens is None:
            ctx_vector = t.zeros(dec_hidden.size())
        else:
            dec_hidden = dec_hidden.unsqueeze(1).repeat(1, pre_dec_hiddens.size(1), 1)   # B, L, 2H

            score = self.attn(dec_hidden, pre_dec_hiddens).squeeze(2)   # B, L

            # softmax

            attention = f.softmax(score, dim=1)

            # context vector

            ctx_vector = t.bmm(attention.unsqueeze(1), pre_dec_hiddens)  # B, 1, L * B, L, 2H  ->  B, 1, 2H
            ctx_vector = ctx_vector.squeeze(1)  # B, 2H

        return ctx_vector
