import torch.nn as nn
import torch.nn.functional as f

from main.common.common import *


class DecoderAttention(nn.Module):

    def __init__(self):
        super(DecoderAttention, self).__init__()

        self.w_attn = nn.Linear(conf('dec-hidden-size'), conf('dec-hidden-size'), False)

    '''
        :params
            dec_hidden          : B, DH
            pre_dec_hiddens     : B, T, DH

        :returns
            ctx_vector          : B, DH
            attention           : B, L
    '''

    def forward(self, dec_hidden, pre_dec_hiddens):
        if pre_dec_hiddens is None:
            ctx_vector = cuda(t.zeros(dec_hidden.size()))
            attention = cuda(t.zeros(dec_hidden.size()))
        else:
            score = t.bmm(self.w_attn(pre_dec_hiddens), dec_hidden.unsqueeze(2)).squeeze(2)  # (B, T, DH) *  (B, DH, 1) => (B, T)

            attention = f.softmax(score, dim=1)  # B, T

            # context vector

            ctx_vector = t.bmm(attention.unsqueeze(1), pre_dec_hiddens).squeeze(1)  # (B, 1, T) * (B, T, DH)  =>  B, DH

        return ctx_vector, attention
