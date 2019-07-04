from main.decoder_attention import *
from main.encoder_attention import *


class Decoder(nn.Module):

    def __init__(self, embedding):
        super(Decoder, self).__init__()

        self.embedding = embedding

        self.emb_size           = conf('emb-size')
        self.enc_hidden_size    = conf('enc-hidden-size')
        self.dec_hidden_size    = conf('dec-hidden-size')
        self.vocab_size         = conf('vocab-size')

        self.intra_dec_attn     = conf('intra-dec-attn')
        self.share_dec_weight   = conf('share-dec-weight')
        self.pointer_generator  = conf('pointer-generator')

        self.y_concat = nn.Linear(2 * self.enc_hidden_size + self.emb_size, self.emb_size)

        self.lstm = nn.LSTMCell(self.emb_size, self.dec_hidden_size)

        self.enc_att = EncoderAttention()

        if self.intra_dec_attn is True:
            self.dec_att = DecoderAttention()

            combined_hidden_size = self.dec_hidden_size + 2 * self.enc_hidden_size + self.dec_hidden_size
        else:
            combined_hidden_size = self.dec_hidden_size + 2 * self.enc_hidden_size

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

    '''
        :params
            dec_input               :   B, E
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
    def forward(self, dec_input,
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

        # decoder hidden state

        y_concat = self.y_concat(t.cat([dec_input, enc_ctx_vector], dim=1))

        dec_hidden, dec_cell = self.lstm(y_concat, (dec_hidden, dec_cell))

        # intra-temporal encoder attention

        enc_ctx_vector, enc_att, enc_temporal_score = self.enc_att(dec_hidden, enc_hiddens, enc_padding_mask, enc_temporal_score, kw)

        # intra-decoder attention

        if self.intra_dec_attn is True:
            dec_ctx_vector, dec_att = self.dec_att(dec_hidden, pre_dec_hiddens)
        else:
            dec_ctx_vector = None
            dec_att = None

        # vocab distribution
        if self.intra_dec_attn is True:
            combined_input = t.cat([dec_hidden, enc_ctx_vector, dec_ctx_vector], dim=1)
        else:
            combined_input = t.cat([dec_hidden, enc_ctx_vector], dim=1)

        vocab_dist = f.softmax(self.vocab_gen(combined_input), dim=1)

        # pointer-generator

        if self.pointer_generator is True:
            ptr_prob = t.sigmoid(self.ptr_gen(combined_input))

            ptr_dist = ptr_prob * enc_att

            vocab_dist = (1 - ptr_prob) * vocab_dist
            vocab_dist = t.cat([vocab_dist, cuda(t.zeros(len(dec_input), max_oov_len))], dim=1)
            vocab_dist.scatter_add(1, extend_vocab_x, ptr_dist)

        return vocab_dist, dec_hidden, dec_cell, enc_ctx_vector, enc_att, enc_temporal_score, dec_ctx_vector, dec_att
