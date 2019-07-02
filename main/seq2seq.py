from main.common.vocab import *
from main.decoder import Decoder
from main.encoder import Encoder
from main.encoder_attention import *
from main.reduce_encoder import ReduceEncoder
from main.kw_encoder import KWEncoder


class Seq2Seq(nn.Module):

    def __init__(self, vocab: Vocab, embedding=None):
        super(Seq2Seq, self).__init__()

        self.emb_size           = conf('emb-size')
        self.enc_hidden_size    = conf('enc-hidden-size')
        self.vocab_size         = conf('vocab-size')

        self.max_dec_steps      = conf('max-dec-steps')

        self.vocab = vocab

        self.embedding = embedding
        if self.embedding is None:
            self.embedding = nn.Embedding(self.vocab_size, self.emb_size , padding_idx=TK_PADDING['id'])

        self.encoder = Encoder()
        self.reduce_encoder = ReduceEncoder()
        self.decoder = Decoder(self.embedding)

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
        batch_size  = len(x)
        max_x_len   = max(x_len)

        x = self.embedding(x)

        enc_outputs, (enc_hidden_n, enc_cell_n) = self.encoder(x, x_len)

        dec_hidden, dec_cell = self.reduce_encoder(enc_hidden_n, enc_cell_n)

        # initial decoder input
        dec_input = cuda(t.tensor([TK_START['id']] * batch_size))

        # encoder padding mask
        enc_padding_mask = t.zeros(batch_size, max_x_len)
        for i in range(batch_size):
            enc_padding_mask[i, :x_len[i]] = t.ones(1, x_len[i])

        enc_padding_mask = cuda(enc_padding_mask)

        # stop decoding mask
        stop_dec_mask = cuda(t.zeros(batch_size))

        # keyword
        if kw is None:
            kw = cuda(t.zeros(batch_size, 1))

        kw = self.kw_encoder(kw.long())

        # initial encoder context vector
        enc_ctx_vector = cuda(t.zeros(batch_size, 2 * self.enc_hidden_size))

        enc_attention = None

        enc_temporal_score = None

        pre_dec_hiddens = None

        y = None

        for i in range(self.max_dec_steps):
            # embedding decoder input
            dec_input = self.embedding(dec_input)

            # decoding
            vocab_dist, dec_hidden, dec_cell, enc_ctx_vector, enc_att, enc_temporal_score, _, _ = self.decoder(
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
            x       : article
            kw      : keyword

        :returns
            y       : summary
            att     : attention
    '''
    def evaluate(self, x, kw):
        self.eval()

        words = x.split()
        x = cuda(t.tensor(self.vocab.words2ids(words) + [TK_STOP['id']]).unsqueeze(0))
        x_len = cuda(t.tensor([len(words) + 1]))

        extend_vocab_x, oov = self.vocab.extend_words2ids(words)
        extend_vocab_x = extend_vocab_x + [TK_STOP['id']]
        extend_vocab_x = cuda(t.tensor(extend_vocab_x).unsqueeze(0))

        max_oov_len = len(oov)

        if kw is not None:
            kw = self.vocab.words2ids(kw if isinstance(kw, (list,)) else kw.split())
            kw = cuda(t.tensor(kw).unsqueeze(0))

        y, att = self.forward(x, x_len, extend_vocab_x, max_oov_len, kw)

        return ' '.join(self.vocab.ids2words(y[0].tolist(), oov)), att[0]
