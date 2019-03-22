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

    def __init__(self, vocab):
        super(Seq2Seq, self).__init__()

        self.emb_size       = conf.get('emb-size')
        self.hidden_size    = conf.get('hidden-size')
        self.max_dec_steps  = conf.get('max-dec-steps')
        self.tf_rate        = conf.get('training')['tf']    # teacher forcing rate

        self.vocab = vocab

        self.embedding = nn.Embedding(self.vocab.size(), self.emb_size)

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.enc_att = EncoderAttention()
        self.dec_att = DecoderAttention()

        self.vocab_gen = nn.Linear(5 * self.hidden_size, self.vocab.size())

        self.p_gen = nn.Linear(5 * self.hidden_size, 1)

    '''
        :param
            x           : B, L, E
            seq_len     : L
            target_y    :
            extend_vocab:
            
        :return
            y
            y_prob
    '''
    def forward(self, x, seq_len, target_y, extend_vocab, extra_zero, teacher_forcing=False, greedy_search=True):
        # embedding input
        x = self.embedding(x)

        # encoding input
        enc_outputs, (enc_hidden_n, enc_cell_n) = self.encoder(x, seq_len)

        # initial decoder input
        dec_input = t.LongTensor(len(enc_outputs)).fill_(self.vocab.word2id(START_DECODING))

        # initial summation of temporal_score
        enc_temporal_score = None

        # previous decoder hidden states
        pre_dec_hiddens = None   # B, T, 2H

        # decoding outputs
        y = []

        # decoding probabilities
        y_prob = []

        for i in range(self.max_dec_steps):

            # embedding decoder input
            dec_input = self.embedding(dec_input)

            # decoding
            vocab_dist, dec_hidden, _, _, enc_temporal_score = self.decode(dec_input, dec_hidden, pre_dec_hiddens, enc_outputs, enc_temporal_score, extend_vocab, extra_zero)

            # output
            if greedy_search:
                _, dec_output = t.max(vocab_dist, dim=1)   # B, 1
            else:
                # sampling
                dec_output = t.multinomial(vocab_dist, 1).squeeze()

            y.append(dec_output)
            y_prob.append(dec_output)

            # stop when reaching STOP_DECODING
            if dec_output == self.vocab.word2id(STOP_DECODING):
                break

            # define next input
            if teacher_forcing:
                use_ground_truth = (t.rand(len(x)) > self.tf).long()
                dec_input = use_ground_truth * target_y[:, i] + (1 - use_ground_truth) * dec_output
            else:
                dec_input = dec_output

            # update previous decoder hidden states
            if pre_dec_hiddens is None:
                pre_dec_hiddens = dec_hidden.unsqueeze(1)
            else:
                pre_dec_hiddens = t.cat([pre_dec_hiddens, dec_hidden.unsqueeze(1)], dim=1)

        return t.tensor(y), t.tensor(y_prob)

    '''
        
    '''
    def decode(self, dec_input, dec_hidden, pre_dec_hiddens, enc_hiddens, enc_temporal_score, extend_vocab, extra_zero, log_prob=True):

        # current decoder hidden
        dec_hidden, _ = self.decoder(dec_input, dec_hidden if pre_dec_hiddens is None else pre_dec_hiddens[:, -1, :])  # B, 2H

        # intra-encoder attention

        # enc_ctx_vector        : B, 2 * H
        # enc_att               : B, L
        # sum_temporal_score    : B, L
        enc_ctx_vector, enc_att, enc_temporal_score = self.enc_att(dec_hidden, enc_hiddens, enc_temporal_score)

        # intra-decoder attention

        dec_ctx_vector = self.dec_att(dec_hidden, pre_dec_hiddens[:, -1, :])  # B, 2*H

        # vocab distribution

        vocab_dist = f.softmax(self.vocab_gen(t.cat([dec_hidden, enc_ctx_vector, dec_ctx_vector])), dim=1)  # B, V

        # pointer-generator

        p_gen = f.sigmoid(self.p_gen(t.cat([dec_hidden, enc_ctx_vector, dec_ctx_vector])))  # B, 1

        # final vocab distribution

        vocab_dist = (1 - p_gen) * vocab_dist  # B, V

        if extra_zero is not None:
            vocab_dist = t.cat([vocab_dist, extra_zero], dim=1)  # B, V + OOV

        p_dist = p_gen * enc_att  # B, L

        vocab_dist = vocab_dist.scatter_add(1, extend_vocab, p_dist)  # B, V + OOV

        if log_prob:
            vocab_dist = t.log(vocab_dist, 1e-31)

        return vocab_dist, dec_hidden, enc_ctx_vector, dec_ctx_vector, enc_temporal_score
