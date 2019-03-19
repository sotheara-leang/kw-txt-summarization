import torch as t
import torch.nn as nn
import torch.nn.functional as f

from main.common.common import *
from main.encoder import Encoder
from main.decoder import Decoder
from main.encoder_attention import EncoderAttention
from main.decoder_attention import DecoderAttention
from main.common.vocab import *


class Seq2Seq(nn.Module):

    def __int__(self, vocab):
        super(Seq2Seq, self).__init__()

        self.emb_size       = conf.get('emb-size')
        self.hidden_size    = conf.get('hidden-size')
        self.vocab_size     = conf.get('vocab-size')
        self.max_dec_steps  = conf.get('max-dec-steps')

        self.vocab = vocab

        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.enc_att = EncoderAttention()
        self.dec_att = DecoderAttention()

        self.vocab_gen = nn.Linear(5 * self.hidden_size, self.vocab_size)

        self.p_gen = nn.Linear(5 * self.hidden_size, 1)

    def forward(self, x, seq_len):

        extra_zeros = {}
        enc_batch_extend_vocab = {}

        # embedding input
        x = self.embedding(x)

        # encoding input
        enc_outputs, (enc_hidden_n, enc_cell_n) = self.encoder(x, seq_len)

        # initial encoder context vector
        enc_ctx_vector = t.zeros(x.size(0), 2 * self.hidden_dim)

        # initial decoder input
        dec_input = t.LongTensor(len(enc_outputs)).fill_(self.vocab.word2id(START_DECODING))

        # initial decoder state
        pre_dec_hidden = enc_hidden_n

        # initial summation of temporal_score
        sum_temporal_score = None

        # decoding output
        y = []

        for i in range(self.max_dec_steps):

            # decode current state
            dec_hidden, _ = self.decoder(dec_input, pre_dec_hidden, enc_ctx_vector)    # B, 2H

            # intra-encoder attention

            # enc_ctx_vector        : B, 2 * H
            # enc_att               : B, L
            # sum_temporal_score    : B, L
            enc_ctx_vector, enc_att, sum_temporal_score = self.enc_att(dec_hidden, enc_hidden_n, sum_temporal_score)

            # intra-decoder attention

            dec_ctx_vector = self.dec_att(dec_hidden, pre_dec_hidden)  # B, 2*H

            # vocab distribution

            vocab_dist = f.softmax(self.vocab_gen(t.cat([dec_hidden, enc_ctx_vector, dec_ctx_vector])), dim=1)  # B, V

            # pointer-generator

            p_gen = f.sigmoid(self.p_gen(t.cat([dec_hidden, enc_ctx_vector, dec_ctx_vector])))  # B, 1

            # final distribution

            vocab_dist = (1 - p_gen) * vocab_dist   # B, V

            if extra_zeros is not None:
                vocab_dist = t.cat([vocab_dist, extra_zeros], dim=1)    # B, V + OOV

            p_dist = p_gen * enc_att    # B, L

            final_vocab_dist = vocab_dist.scatter_add(1, enc_batch_extend_vocab, p_dist)    # B, V + OOV

            _, dec_input = t.max(final_vocab_dist, dim=1)

            pre_dec_hidden = dec_hidden

            # store final output
            y.append(dec_input)

            # reach the token STOP_DECODING
            if dec_input == self.vocab.word2id(STOP_DECODING):
                break

        return y
