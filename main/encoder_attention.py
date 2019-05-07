import torch.nn as nn
import torch.nn.functional as f

from main.common.common import *


class EncoderAttention(nn.Module):

    def __init__(self):
        super(EncoderAttention, self).__init__()

        self.w_e = nn.Linear(2 * conf.get('enc-hidden-size'), conf.get('dec-hidden-size'), False)
        self.w_d = nn.Linear(conf.get('dec-hidden-size'), conf.get('dec-hidden-size'), False)
        self.w_k = nn.Linear(conf.get('emb-size'), conf.get('dec-hidden-size'))

        self.v = nn.Linear(conf.get('dec-hidden-size'), 1, False)

    '''
        :params
            dec_hidden          : B, DH
            enc_hiddens         : B, L, EH
            enc_padding_mask    : B, L
            enc_temporal_score  : B, L
            kw                  : B, E

        :returns
            ctx_vector          : B, EH
            attention           : B, L
            enc_temporal_score  : B, L
    '''

    def forward(self, dec_hidden, enc_hiddens, enc_padding_mask, enc_temporal_score, kw):
        enc_out = self.w_e(enc_hiddens)                 # B, L, DH

        dec_out = self.w_d(dec_hidden).unsqueeze(1)     # B, 1, DH

        kw_out = self.w_k(kw).unsqueeze(1)              # B, 1, DH

        score = self.v(t.tanh(dec_out + enc_out + kw_out)).squeeze(2)  # B, L

        score = score.masked_fill_(enc_padding_mask, -float('inf'))

        score = f.softmax(score, dim=1)

        # temporal normalization

        if enc_temporal_score is None:
            enc_temporal_score = score
        else:
            score = score / (enc_temporal_score + 1e-10)
            enc_temporal_score = enc_temporal_score + score

        # normalization

        attention = score / (t.sum(score, dim=1).unsqueeze(1) + 1e-10)  # B, L

        # context vector

        ctx_vector = t.bmm(attention.unsqueeze(1), enc_hiddens).squeeze(1)  # (B, 1, L) * (B, L, EH)  =>  B, EH

        return ctx_vector, attention, enc_temporal_score
