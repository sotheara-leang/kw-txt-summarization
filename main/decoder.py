import torch as t
import torch.nn as nn
import torch.nn.functional as f

from main.common.common import *
from main.encoder_attention import EncoderAttention
from main.decoder_attention import DecoderAttention


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.enc_attr = EncoderAttention()
        self.dec_attr = DecoderAttention()

        emb_size = conf.get('emb-size')
        hidden_size = conf.get('hidden-size')
        vocab_size = conf.get('vocab-size')

        self.x_concat = nn.Linear(2 * hidden_size + emb_size, emb_size)

        self.lstm = nn.LSTMCell(emb_size, 2 * hidden_size)

        self.vocab_gen = nn.Linear(5 * hidden_size, vocab_size)

        self.p_gen = nn.Linear(5 * hidden_size, 1)

    '''
        :param
            x               : B, 2H
            pre_hidden  : B, 2H
            enc_hidden      : B, L, 2H
            ctx_vector      : B, 2H
            
        :return
           
    '''

    def forward(self, x, pre_hidden, enc_hidden, ctx_vector, sum_temporal_score):
        x = self.x_concat(t.cat([x, ctx_vector], dim=1))    # B, E + 2H

        # new decoder state
        hidden, _ = self.lstm(x, pre_hidden)    # B, 2H

        # intra-encoder attention

        # enc_ctx_vector        : B, 2 * H
        # sum_temporal_score    : B, L
        enc_ctx_vector, enc_attr, sum_temporal_score = self.enc_attr(hidden, enc_hidden, sum_temporal_score)

        # intra-decoder attention

        dec_ctx_vector = self.dec_attr(hidden, pre_hidden)     # B, 2*H

        # vocab distribution

        vocab_dist = f.softmax(self.vocab_gen(t.cat([hidden, enc_ctx_vector, dec_ctx_vector])), dim=1)  # B, V

        # pointer-generator

        p_gen = f.sigmoid(self.p_gen(t.cat([hidden, enc_ctx_vector, dec_ctx_vector])))  # B, 1

        #
        vocab_dist = (1 - p_gen) * vocab_dist

        p_dist = p_gen * enc_attr

        # final distribution

        return hidden, sum_temporal_score
