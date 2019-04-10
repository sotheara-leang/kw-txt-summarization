import torch.nn as nn
import torch.nn.functional as f

from main.encoder import Encoder
from main.decoder import Decoder
from main.encoder_attention import EncoderAttention
from main.decoder_attention import DecoderAttention
from main.common.vocab import *
from main.common.common import *


class Seq2Seq(nn.Module):

    def __init__(self, vocab: Vocab, embedding=None):
        super(Seq2Seq, self).__init__()

        self.emb_size               = conf.get('emb-size')
        self.hidden_size            = conf.get('hidden-size')
        self.vocab_size             = conf.get('vocab-size')
        self.max_dec_steps          = conf.get('max-dec-steps')
        self.sharing_decoder_weight = conf.get('sharing-decoder-weight')

        self.vocab = vocab

        self.embedding = embedding
        if self.embedding is None:
            self.embedding = nn.Embedding(self.vocab.size(), self.emb_size, padding_idx=TK_PADDING['id'])

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.enc_att = EncoderAttention()
        self.dec_att = DecoderAttention()

        self.ptr_gen = nn.Linear(6 * self.hidden_size, 1)

        # sharing decoder weight
        if self.sharing_decoder_weight is True:
            projection_layer = nn.Linear(6 * self.hidden_size, self.emb_size)

            output_layer = nn.Linear(self.emb_size, self.vocab_size)
            output_layer.weight = self.embedding.weight     # sharing weight with embedding

            self.vocab_gen = nn.Sequential(
                projection_layer,
                output_layer
            )
        else:
            self.vocab_gen = nn.Linear(6 * self.hidden_size, self.vocab.size())

    '''
            :params
                x                : B, L
                x_len            : L
                extend_vocab_x   : B, V + OOV
                oov              : B, L

            :returns
                y                : B, L
    '''
    def forward(self, x, x_len, extend_vocab_x, oov):
        batch_size = len(x)

        x = self.embedding(x)  # B, L, E

        enc_outputs, (enc_hidden_n, enc_cell_n) = self.encoder(x, x_len)  # (B, L, 2H) , (B, 2H)

        # initial decoder input = START_DECODING
        dec_input = cuda(t.tensor([TK_START['id']] * batch_size))  # B

        # initial decoder hidden = encoder last hidden
        dec_hidden = enc_hidden_n

        # initial decoder cell = encoder last cell
        dec_cell = enc_cell_n

        # encoder temporal attention score
        enc_temporal_score = None   # B, L

        pre_dec_hiddens = None  # B, T, 2H

        y = None  # B, L

        #
        enc_padding_mask = t.zeros(batch_size, max(x_len))
        for i in range(batch_size):
            enc_padding_mask[i, :x_len[i]] = t.ones(1, x_len[i])

        # stop decoding mask
        stop_dec_mask = cuda(t.zeros(len(x)))

        #
        max_oov_len = max([len(vocab) for vocab in oov])

        for i in range(self.max_dec_steps):
            # decoding
            vocab_dist, dec_hidden, dec_cell, enc_ctx_vector, _, enc_temporal_score = self.decode(
                dec_input,
                dec_hidden,
                dec_cell,
                pre_dec_hiddens,
                enc_outputs,
                enc_padding_mask,
                enc_temporal_score,
                extend_vocab_x,
                max_oov_len)

            _, dec_output = t.max(vocab_dist, dim=1)  # B - word idx

            y = dec_output.unsqueeze(1) if y is None else t.cat([y, dec_output.unsqueeze(1)], dim=1)

            # set mask = 1 If output is [STOP]
            stop_dec_mask[(stop_dec_mask == 0) + (dec_output == TK_STOP['id']) == 2] = 1

            # stop when all masks are 1
            if len(stop_dec_mask[stop_dec_mask == 1]) == len(stop_dec_mask):
                break

            pre_dec_hiddens = dec_hidden.unsqueeze(1) if pre_dec_hiddens is None else t.cat([pre_dec_hiddens, dec_hidden.unsqueeze(1)], dim=1)

            dec_input = dec_output

        return y

    '''
        :params
            dec_input               :   B
            dec_hidden              :   B, 2H
            dec_cell                :   B, 2H
            pre_dec_hiddens         :   B, T, 2H
            enc_hiddens             :   B, L, 2H
            enc_padding_mask        :   B, L
            enc_temporal_score      :   B, L
            extend_vocab_x          :   B, V + OOV
            max_oov_len             :   C
            
        :returns
            vocab_dist          :   B, V + OOV
            dec_hidden          :   B, 2H
            enc_ctx_vector      :   B, 2H
            dec_ctx_vector      :   B, 2H
            enc_temporal_score  :   B, L
    '''
    def decode(self, dec_input,
               dec_hidden,
               dec_cell,
               pre_dec_hiddens,
               enc_hiddens,
               enc_padding_mask,
               enc_temporal_score,
               extend_vocab_x,
               max_oov_len):

        # embedding input
        dec_input = self.embedding(dec_input)   # B, E

        # current hidden & cell
        dec_hidden, dec_cell = self.decoder(dec_input, dec_hidden if pre_dec_hiddens is None else pre_dec_hiddens[:, -1, :], dec_cell)  # B, 2H

        # intra-temporal encoder attention

        # enc_ctx_vector        : B, 2H
        # enc_att               : B, L
        # sum_temporal_score    : B, L
        enc_ctx_vector, enc_att, enc_temporal_score = self.enc_att(dec_hidden, enc_hiddens, enc_padding_mask, enc_temporal_score)

        # intra-decoder attention

        dec_ctx_vector = self.dec_att(dec_hidden, pre_dec_hiddens)  # B, 2H

        # pointer-generator

        combine_input = t.cat([dec_hidden, enc_ctx_vector, dec_ctx_vector], dim=1)

        ptr_prob = t.sigmoid(self.ptr_gen(combine_input))  # B

        ptr_dist = ptr_prob * enc_att  # B, L

        # vocab distribution

        vocab_dist = f.softmax(self.vocab_gen(combine_input), dim=1)  # B, V
        vocab_dist = (1 - ptr_prob) * vocab_dist

        # final vocab distribution

        final_vocab_dist = cuda(t.zeros(len(dec_input), self.vocab.size() + max_oov_len))     # B, V + OOV
        final_vocab_dist[:, :self.vocab.size()] = vocab_dist
        final_vocab_dist.scatter_add(1, extend_vocab_x, ptr_dist)

        return final_vocab_dist, dec_hidden, dec_cell, enc_ctx_vector, dec_ctx_vector, enc_temporal_score

    '''
        :params
            x       :
        :returns
            y       :
    '''
    def summarize(self, x):
        words = x.split()

        x = cuda(t.tensor(self.vocab.words2ids(words) + [TK_STOP['id']]).unsqueeze(0))
        x_len = cuda(t.tensor([len(words) + 1]))

        extend_vocab_x, oov = self.vocab.extend_words2ids(words)
        extend_vocab_x = cuda(t.tensor(extend_vocab_x).unsqueeze(0))
        oovs = [oov]

        y = self.forward(x, x_len, extend_vocab_x, oovs)[0].squeeze(0)

        return ' '.join(self.vocab.ids2words(y.tolist(), oov))
