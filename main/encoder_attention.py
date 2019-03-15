import torch as t
import torch.nn as nn
import torch.nn.functional as f

from main.common.common import *


class EncoderAttention(nn.Module):

    def __init__(self):
        super(EncoderAttention, self).__init__()

        self.attn = nn.Bilinear(2 * conf.get('hidden-size'), 2 * conf.get('hidden-size'), 1, False)

    '''
        :param
            dec_hidden  : B, 2H
            enc_hidden  : B, L, 2H
            sum_score   : B, L
        
        :return
            context_vector  : B, 2*H
            sum_score       : B, L
    '''
    def forward(self, dec_hidden, enc_hidden, sum_temporal_score):
        dec_hidden = dec_hidden.unsqueeze(1).repeat(1, enc_hidden.size(1), 1)   # B, L, 2H

        score = self.attn(dec_hidden, enc_hidden).squeeze(2)   # B, L

        # temporal normalization

        exp_score = t.exp(score)    # B, L
        if sum_temporal_score is None:
            score = exp_score
            sum_temporal_score = exp_score
        else:
            score = exp_score / sum_temporal_score
            sum_temporal_score += exp_score

        # normalization

        normalization_factor = score.sum(1, keepdim=True)   # B, L
        attention = score / normalization_factor            # B, L

        # context vector

        context_vector = t.bmm(attention.unsqueeze(1), enc_hidden)  # B, 1, L * B, L, 2H  ->  B, 1, 2*H
        context_vector = context_vector.squeeze(1)  # B, 2*H

        return context_vector, attention, sum_temporal_score
