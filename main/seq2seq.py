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

        self.vocab_gen = nn.Linear(6 * self.hidden_size, self.vocab.size())

        self.p_gen = nn.Linear(6 * self.hidden_size, 1)

    '''
        :params
            x               : B, L
            seq_len         : L
            target_y        : B, L
            extend_vocab    : C
            teacher_forcing
            greedy_search
            
        :returns
            y
            y_prob
    '''
    def forward(self, x, seq_len, target_y, extend_vocab, max_ovv_len, teacher_forcing=False, greedy_search=True):
        # embedding input
        x = self.embedding(x)

        # encoding input
        enc_outputs, (enc_hidden_n, _) = self.encoder(x, seq_len)

        # initial decoder input = START_DECODING
        dec_input = t.tensor([self.vocab.word2id(START_DECODING)] * x.size(0))

        # initial decoder hidden = encoder last hidden
        dec_hidden = enc_hidden_n

        # initial summation of temporal_score
        enc_temporal_score = None

        # previous decoder hidden states
        pre_dec_hiddens = None   # B, T, 2H

        # decoding outputs
        y = None

        # decoding probabilities
        y_prob = None

        # decoding length
        if target_y is None:
            decode_len = self.max_dec_steps
        else:
            decode_len = target_y.size(1)

        for i in range(decode_len):

            # decoding
            vocab_dist, dec_hidden, _, _, enc_temporal_score = self.decode(dec_input, dec_hidden, pre_dec_hiddens, enc_outputs, enc_temporal_score, extend_vocab, max_ovv_len)

            # output
            if greedy_search:
                _, dec_output = t.max(vocab_dist, dim=1)   # B, 1
            else:
                # sampling
                dec_output = t.multinomial(vocab_dist, 1).squeeze()     # B

            # update output
            y = dec_output.unsqueeze(1) if y is None else t.cat([y, dec_output.unsqueeze(1)], dim=1)    # B
            y_prob = vocab_dist.unsqueeze(1) if y_prob is None else t.cat([y_prob, vocab_dist.unsqueeze(1)], dim=1) #B

            # update previous decoder hidden states
            pre_dec_hiddens = dec_hidden.unsqueeze(1) if pre_dec_hiddens is None else t.cat([pre_dec_hiddens, dec_hidden.unsqueeze(1)], dim=1)

            # define next input
            if teacher_forcing:
                use_ground_truth = (t.rand(len(x)) > self.tf_rate).long()  # B
                dec_input = use_ground_truth * target_y[:, i] + (1 - use_ground_truth) * dec_output
            else:
                dec_input = dec_output

            #
            is_oov = (dec_input >= self.vocab.size()).long()
            dec_input = (1 - is_oov) * dec_input + is_oov * self.vocab.word2id(UNKNOWN_TOKEN)

        return y, y_prob

    '''
        :params
            dec_input           :  B
            dec_hidden
            pre_dec_hiddens
            enc_hiddens
            
        :returns
            
    '''
    def decode(self, dec_input, dec_hidden, pre_dec_hiddens, enc_hiddens, enc_temporal_score, extend_vocab, max_ovv_len, log_prob=True):
        # embedding decoder input
        dec_input = self.embedding(dec_input)

        # current decoder hidden
        dec_hidden = self.decoder(dec_input, dec_hidden if pre_dec_hiddens is None else pre_dec_hiddens[:, -1, :])  # B, 2H

        # intra-encoder attention

        # enc_ctx_vector        : B, 2 * H
        # enc_att               : B, L
        # sum_temporal_score    : B, L
        enc_ctx_vector, enc_att, enc_temporal_score = self.enc_att(dec_hidden, enc_hiddens, enc_temporal_score)

        # intra-decoder attention

        dec_ctx_vector = self.dec_att(dec_hidden, pre_dec_hiddens)  # B, 2H

        # pointer-generator

        ptr_gen = t.sigmoid(self.p_gen(t.cat([dec_hidden, enc_ctx_vector, dec_ctx_vector], dim=1)))  # B, 1

        # pointer distribution

        ptr_dist = ptr_gen * enc_att  # B, L

        # vocab distribution

        vocab_dist = f.softmax(self.vocab_gen(t.cat([dec_hidden, enc_ctx_vector, dec_ctx_vector], dim=1)), dim=1)  # B, V
        vocab_dist = (1 - ptr_gen) * vocab_dist  # B, V

        # final vocab distribution
        extend_vocab_dist = t.zeros(len(dec_input), self.vocab.size() + max_ovv_len)     # B, V + OOV
        #
        extend_vocab_dist[:, :self.vocab.size()] = vocab_dist
        #
        extend_vocab_dist.scatter_add(1, extend_vocab, ptr_dist)

        if log_prob:
            extend_vocab_dist = t.log(extend_vocab_dist + 1e-31)

        return extend_vocab_dist, dec_hidden, enc_ctx_vector, dec_ctx_vector, enc_temporal_score
