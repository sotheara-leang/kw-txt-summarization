import torch as t
import torch.nn as nn
import torch.nn.functional as f

from main.common.common import *


class DecoderAttention(nn.Module):

    def __init__(self):
        super(DecoderAttention, self).__init__()

        self.attn = nn.Bilinear(2 * conf.get('hidden-size'), 2 * conf.get('hidden-size'), 1, False)

    '''
        :params
            dec_hidden       : B, 2H
            pre_dec_hiddens  : B, T, 2H
            
        :returns
            ctx_vector   : B, 2*H
    '''
    def forward(self, dec_hidden, pre_dec_hiddens):
        if pre_dec_hiddens is None:
            ctx_vector = cuda(t.zeros(dec_hidden.size()))
        else:
            dec_hidden = dec_hidden.unsqueeze(1).expand(-1, pre_dec_hiddens.size(1), -1).contiguous() # B, T, 2H

            score = self.attn(dec_hidden, pre_dec_hiddens).squeeze(2)   # B, T

            # softmax

            attention = f.softmax(score, dim=1)  # B, T

            # context vector

            ctx_vector = t.bmm(attention.unsqueeze(1), pre_dec_hiddens)  # B, 1, T * B, T, 2H  =>  B, 1, 2H
            ctx_vector = ctx_vector.squeeze(1)  # B, 2H

        return ctx_vector
