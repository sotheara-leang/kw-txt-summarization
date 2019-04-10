import torch.nn as nn

from main.common.common import *


class EncoderAttention(nn.Module):

    def __init__(self):
        super(EncoderAttention, self).__init__()

        self.enc = nn.Linear(2 * conf.get('hidden-size'), 2 * conf.get('hidden-size'), False)
        self.dec = nn.Linear(2 * conf.get('hidden-size'), 2 * conf.get('hidden-size'), False)
        self.kw = nn.Linear(conf.get('emb-size'), 2 * conf.get('hidden-size'))

        self.v = nn.Linear(2 * conf.get('hidden-size'), 1)

    '''
        :params
            dec_hidden          : B, 2H
            enc_hiddens         : B, L, 2H
            enc_temporal_score  : B, L
            enc_padding_mask    : B, L
            enc_temporal_score  : B, L
            kw                  : B, N
            
        :returns
            ctx_vector          : B, 2H
            att_dist            : B, L
            enc_temporal_score  : B, L
    '''
    def forward(self, dec_hidden, enc_hiddens, enc_padding_mask, enc_temporal_score, kw):
        enc_out = self.enc(enc_hiddens)

        dec_out = self.dec(dec_hidden).unsqueeze(1)  # B, 1, 2H

        kw_out = self.kw(kw).unsqueeze(1)            # B, 1, 2H

        score = self.v(t.tanh(dec_out + enc_out + kw_out)).squeeze(2)   # B, L

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
