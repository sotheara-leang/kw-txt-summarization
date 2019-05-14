import torch.nn as nn
import torch.nn.functional as f

from main.encoder import Encoder
from main.reduce_encoder import ReduceEncoder
from main.decoder import Decoder
from main.encoder_attention import *
from main.decoder_attention import DecoderAttention
from main.kw_encoder import KWEncoder
from main.common.vocab import *
from main.common.common import *


class Seq2Seq(nn.Module):

    def __init__(self, vocab: Vocab, embedding=None):
        super(Seq2Seq, self).__init__()

        self.emb_size           = conf('emb-size')
        self.enc_hidden_size    = conf('enc-hidden-size')
        self.dec_hidden_size    = conf('dec-hidden-size')
        self.vocab_size         = conf('vocab-size')
        self.max_dec_steps      = conf('max-dec-steps')
        self.share_dec_weight   = conf('share-dec-weight')
        self.pointer_generator  = conf('pointer-generator')

        self.vocab = vocab

        self.embedding = embedding
        if self.embedding is None:
            self.embedding = nn.Embedding(self.vocab.size(), self.emb_size, padding_idx=TK_PADDING['id'])

        self.encoder = Encoder()
        self.reduce_encoder = ReduceEncoder()
        self.decoder = Decoder()

        self.enc_att = EncoderAttention()
        self.dec_att = DecoderAttention()

        combined_hidden_size = self.dec_hidden_size + 2 * self.enc_hidden_size + self.dec_hidden_size

        if self.pointer_generator is True:
            self.ptr_gen = nn.Linear(combined_hidden_size, 1)

        # sharing decoder weight
        if self.share_dec_weight is True:
            proj_layer = nn.Linear(combined_hidden_size, self.emb_size)

            output_layer = nn.Linear(self.emb_size, self.vocab_size)
            output_layer.weight.data = self.embedding.weight.data  # sharing weight with embedding

            self.vocab_gen = nn.Sequential(
                proj_layer,
                output_layer
            )
        else:
            self.vocab_gen = nn.Linear(combined_hidden_size, self.vocab_size)

        self.kw_encoder = KWEncoder(self.embedding)

    '''
            :params
                x                : B, L
                x_len            : B
                extend_vocab_x   : B, L
                max_oov_len      : C
                kw               : B, L

            :returns
                y                : B, L
                att              : B, L
    '''

    def forward(self, x, x_len, extend_vocab_x, max_oov_len, kw):
        batch_size = len(x)

        x = self.embedding(x)

        enc_outputs, (enc_hidden_n, enc_cell_n) = self.encoder(x, x_len)

        enc_hidden_n, enc_cell_n = self.reduce_encoder(enc_hidden_n, enc_cell_n)

        # initial decoder hidden
        dec_hidden = enc_hidden_n

        # initial decoder cell
        dec_cell = cuda(t.zeros(batch_size, self.dec_hidden_size))

        # initial decoder input
        dec_input = cuda(t.tensor([TK_START['id']] * batch_size))

        # encoder padding mask
        enc_padding_mask = t.zeros(batch_size, max(x_len))
        for i in range(batch_size):
            enc_padding_mask[i, :x_len[i]] = t.ones(1, x_len[i])

        enc_padding_mask = cuda(enc_padding_mask)

        # stop decoding mask
        stop_dec_mask = cuda(t.zeros(batch_size))

        # keyword
        kw = self.kw_encoder(kw)

        enc_ctx_vector = cuda(t.zeros(batch_size, 2 * self.enc_hidden_size))

        enc_attention = None

        enc_temporal_score = None

        pre_dec_hiddens = None

        y = None

        for i in range(self.max_dec_steps):
            # decoding
            vocab_dist, dec_hidden, dec_cell, enc_ctx_vector, enc_att, enc_temporal_score, _, _ = self.decode(
                dec_input,
                dec_hidden,
                dec_cell,
                pre_dec_hiddens,
                enc_outputs,
                enc_padding_mask,
                enc_temporal_score,
                enc_ctx_vector,
                extend_vocab_x,
                max_oov_len,
                kw)

            enc_attention = enc_att.unsqueeze(1).detach() if enc_attention is None else t.cat([enc_attention, enc_att.unsqueeze(1).detach()], dim=1)

            ## output

            dec_output = t.max(vocab_dist, dim=1)[1].detach()

            y = dec_output.unsqueeze(1) if y is None else t.cat([y, dec_output.unsqueeze(1)], dim=1)

            ## stop decoding mask

            stop_dec_mask[dec_output == TK_STOP['id']] = 1

            if len(stop_dec_mask[stop_dec_mask == 1]) == len(stop_dec_mask):
                break

            pre_dec_hiddens = dec_hidden.unsqueeze(1) if pre_dec_hiddens is None else t.cat([pre_dec_hiddens, dec_hidden.unsqueeze(1)], dim=1)

            dec_input = dec_output

        return y, enc_attention

    '''
        :params
            dec_input               :   B
            dec_hidden              :   B, DH
            dec_cell                :   B, DH
            pre_dec_hiddens         :   B, T, DH
            enc_hiddens             :   B, L, EH
            enc_padding_mask        :   B, L
            enc_temporal_score      :   B, L
            enc_ctx_vector          :   B, 2EH
            extend_vocab_x          :   B, L
            max_oov_len             :   C
            kw                      :   B, E

        :returns
            vocab_dist              :   B, V + OOV
            dec_hidden              :   B, DH
            dec_cell                :   B, DH
            enc_ctx_vector          :   B, 2EH
            enc_attention           :   B, L
            enc_temporal_score      :   B, L
            dec_ctx_vector          :   B, DH
            dec_attention           :   B, L
    '''

    def decode(self, dec_input,
               dec_hidden,
               dec_cell,
               pre_dec_hiddens,
               enc_hiddens,
               enc_padding_mask,
               enc_temporal_score,
               enc_ctx_vector,
               extend_vocab_x,
               max_oov_len,
               kw):

        dec_input = self.embedding(dec_input)

        dec_hidden, dec_cell = self.decoder(dec_input, dec_hidden, dec_cell, enc_ctx_vector)

        # intra-temporal encoder attention

        enc_ctx_vector, enc_att, enc_temporal_score = self.enc_att(dec_hidden, enc_hiddens, enc_padding_mask, enc_temporal_score, kw)

        # intra-decoder attention

        dec_ctx_vector, dec_att = self.dec_att(dec_hidden, pre_dec_hiddens)

        # vocab distribution

        combined_input = t.cat([dec_hidden, enc_ctx_vector, dec_ctx_vector], dim=1)

        vocab_dist = f.softmax(self.vocab_gen(combined_input), dim=1)

        # pointer-generator

        if self.pointer_generator is True:
            ptr_prob = t.sigmoid(self.ptr_gen(combined_input))

            ptr_dist = ptr_prob * enc_att

            vocab_dist = (1 - ptr_prob) * vocab_dist

            final_vocab_dist = cuda(t.zeros(len(dec_input), self.vocab.size() + max_oov_len))
            final_vocab_dist[:, :self.vocab.size()] = vocab_dist
            final_vocab_dist.scatter_add(1, extend_vocab_x, ptr_dist)
        else:
            final_vocab_dist = vocab_dist

        return final_vocab_dist, dec_hidden, dec_cell, enc_ctx_vector, enc_att, enc_temporal_score, dec_ctx_vector, dec_att

    '''
        :params
            x       : article
            kw      : keyword

        :returns
            y       : summary
            att     : attention
    '''

    def evaluate(self, x):
        self.eval()

        words = x.split()

        x = cuda(t.tensor(self.vocab.words2ids(words) + [TK_STOP['id']]).unsqueeze(0))
        x_len = cuda(t.tensor([len(words) + 1]))

        extend_vocab_x, oov = self.vocab.extend_words2ids(words)

        extend_vocab_x = extend_vocab_x + [TK_STOP['id']]
        extend_vocab_x = cuda(t.tensor(extend_vocab_x).unsqueeze(0))

        max_oov_len = len(oov)

        kw = self.vocab.words2ids(kw.split())

        y, att = self.forward(x, x_len, extend_vocab_x, max_oov_len, kw)

        return ' '.join(self.vocab.ids2words(y[0].tolist(), oov)), att[0]
