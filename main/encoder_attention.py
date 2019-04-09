import torch.nn as nn

from main.common.common import *


class EncoderAttention(nn.Module):

    def __init__(self):
        super(EncoderAttention, self).__init__()

        self.attn = nn.Bilinear(2 * conf.get('hidden-size'), 2 * conf.get('hidden-size'), 1, False)

    '''
        :params
            dec_hidden          : B, 2H
            enc_hiddens         : B, L, 2H
            enc_temporal_score  : B, L
            enc_padding_mask    : B, L
        
        :returns
            ctx_vector          : B, 2H
            att_dist            : B, L
            enc_temporal_score  : B, L
    '''
    def forward(self, dec_hidden, enc_hiddens, enc_padding_mask, enc_temporal_score):
        dec_hidden = dec_hidden.unsqueeze(1).expand(-1, enc_hiddens.size(1), -1).contiguous()  # B, L, 2H

        score = self.attn(dec_hidden, enc_hiddens).squeeze(2)   # B, L

        # temporal normalization

        exp_score = t.exp(score)    # B, L
        if enc_temporal_score is None:
            score = exp_score
            enc_temporal_score = cuda(t.zeros(score.size()).fill_(1e-10)) + exp_score
        else:
            score = exp_score / enc_temporal_score
            enc_temporal_score = enc_temporal_score + exp_score

        # masking

        score = score * enc_padding_mask.float()

        # normalization

        normalization_factor = score.sum(1, keepdim=True)   # B, L
        attention = score / normalization_factor  # B, L

        # context vector

        ctx_vector = t.bmm(attention.unsqueeze(1), enc_hiddens)  # B, 1, L * B, L, 2H  =>  B, 1, 2H
        ctx_vector = ctx_vector.squeeze(1)  # B, 2*H

        return ctx_vector, attention, enc_temporal_score
