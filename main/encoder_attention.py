import torch as t
import torch.nn as nn
import torch.nn.functional as f

from main.common.common import *


class EncoderAttention(nn.Module):

    def __init__(self):
        super(EncoderAttention, self).__init__()

        self.attn = nn.Bilinear(2 * conf.get('hidden-size'), conf.get('hidden-size'), 1, False)

    '''
        dec_hidden  : B, 2H
        enc_hidden  : B, L, 2H
        sum_score   : B, 1
    '''
    def forward(self, dec_hidden, enc_hidden, sum_score):
        score = self.attn(dec_hidden.unsqueeze(1), enc_hidden)   # B, L, 1

        # normalized score

        exp_score = t.exp(score)    # B, L, 1
        if sum_score is None:
            score = exp_score
            sum_score = exp_score
        else:
            score = exp_score / sum_score
            sum_score += exp_score

        # softmax

        attention = f.softmax(score, dim=1)     # B, 1

        # context vector

        context_vector = t.bmm(attention.unsqueeze(1), enc_hidden)  # B, 1, L * B, L, 2H  ->  B, 1, 2*H
        context_vector = context_vector.squeeze(1)  # B, 2*H

        return context_vector, sum_score
