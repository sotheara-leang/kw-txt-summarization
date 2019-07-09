import torch.nn as nn

from main.common.common import *


class EncoderAttention(nn.Module):

    def __init__(self):
        super(EncoderAttention, self).__init__()

        self.w_e = nn.Linear(2 * conf('enc-hidden-size'), conf('dec-hidden-size'), False)
        self.w_d = nn.Linear(conf('dec-hidden-size'), conf('dec-hidden-size'), False)
        self.w_k = nn.Linear(conf('emb-size'), conf('dec-hidden-size'))

        self.v = nn.Linear(conf('dec-hidden-size'), 1, False)

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

        # temporal normalization

        exp_score = t.exp(score)

        if enc_temporal_score is None:
            score = exp_score
            enc_temporal_score = t.clamp(exp_score, min=1e-10)
        else:
            score = exp_score / enc_temporal_score
            enc_temporal_score = enc_temporal_score + exp_score

        # normalization

        score = score * enc_padding_mask.float()

        factor = t.sum(score, dim=1)
        factor = t.clamp(factor, min=1e-10)

        attention = score / factor.unsqueeze(1)  # B, L

        # context vector

        ctx_vector = t.bmm(attention.unsqueeze(1), enc_hiddens).squeeze(1)  # (B, 1, L) * (B, L, EH)  =>  B, EH

        return ctx_vector, attention, enc_temporal_score
