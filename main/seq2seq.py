import torch.nn as nn
import torch.nn.functional as f

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
        self.tf_rate        = conf.get('train:tf')    # teacher forcing rate

        self.vocab = vocab

        self.embedding = nn.Embedding(self.vocab.size(), self.emb_size, padding_idx=TK_PADDING.idx)

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
            teacher_forcing : False
            greedy_search   : True
            
        :returns
            y               : B, L
            loss            : B, L
    '''
    def forward(self, x, seq_len,
                target_y,
                extend_vocab,
                max_ovv_len,
                calculate_loss=False,
                teacher_forcing=False,
                greedy_search=True):

        # embedding input
        x = self.embedding(x)   # B, L, E

        # encoding input
        enc_outputs, (enc_hidden_n, _) = self.encoder(x, seq_len)   # B, L, 2H, B, 2H

        # initial decoder input = START_DECODING
        dec_input = cuda(t.tensor([TK_START_DECODING.idx] * x.size(0)))  # B

        # initial decoder hidden = encoder last hidden
        dec_hidden = enc_hidden_n

        # intra-encoder attention score from previous time step
        enc_temporal_score = None

        #
        pre_dec_hiddens = None   # B, T, 2H

        # output
        y = None    # B, L

        # total loss
        loss = None   # B, L

        #
        dec_len = self.max_dec_steps if target_y is None else target_y.size(1)

        for i in range(dec_len):

            # decoding
            vocab_dist, dec_hidden, _, _, enc_temporal_score = self.decode(
                dec_input,
                dec_hidden,
                pre_dec_hiddens,
                enc_outputs,
                enc_temporal_score,
                extend_vocab,
                max_ovv_len)

            # define output from vocab distribution
            if greedy_search:
                _, dec_output = t.max(vocab_dist, dim=1)   # B - word idx
            else:
                # sampling
                dec_output = t.multinomial(vocab_dist, 1).squeeze()     # B - word idx

            # record output
            y = dec_output.unsqueeze(1) if y is None else t.cat([y, dec_output.unsqueeze(1)], dim=1)    # B, L

            # calculate loss
            if calculate_loss and target_y is not None:
                step_loss = f.nll_loss(t.log(vocab_dist + 1e-12), target_y[:, i], reduction='none', ignore_index=TK_PADDING.idx)

                loss = step_loss.unsqueeze(1) if loss is None else t.cat([loss, step_loss.unsqueeze(1)], dim=1)    # B, L

            # record decoder hidden
            pre_dec_hiddens = dec_hidden.unsqueeze(1) if pre_dec_hiddens is None else t.cat([pre_dec_hiddens, dec_hidden.unsqueeze(1)], dim=1)

            # define next input
            if teacher_forcing and target_y is not None:
                use_ground_truth = t.rand(len(x)) > self.tf_rate  # B
                use_ground_truth = cuda(use_ground_truth.long())

                dec_input = use_ground_truth * target_y[:, i] + (1 - use_ground_truth) * dec_output     # B
            else:
                dec_input = dec_output

            # if next input is oov, change it to UNKNOWN_TOKEN
            is_oov = (dec_input >= self.vocab.size()).long()
            dec_input = (1 - is_oov) * dec_input + is_oov * TK_UNKNOWN.idx

        return y, loss

    '''
        :params
            dec_input           :   B
            dec_hidden          :   B, 2H
            pre_dec_hiddens     :   B, T, 2H
            enc_hiddens         :   B, L, 2H
            enc_temporal_score  :   C
            extend_vocab        :
            max_ovv_len         :   C
            
        :returns
            final_vocab_dist    :   B, V + OOV
            dec_hidden          :   B, 2H
            enc_ctx_vector      :   B, 2H
            dec_ctx_vector      :   B, 2H
            enc_temporal_score  :   C
    '''
    def decode(self, dec_input,
               dec_hidden,
               pre_dec_hiddens,
               enc_hiddens,
               enc_temporal_score,
               extend_vocab,
               max_ovv_len):

        # embedding input
        dec_input = self.embedding(dec_input)   # B, E

        # current hidden
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

        final_vocab_dist = cuda(t.zeros(len(dec_input), self.vocab.size() + max_ovv_len))     # B, V + OOV
        #
        final_vocab_dist[:, :self.vocab.size()] = vocab_dist
        #
        final_vocab_dist.scatter_add(1, extend_vocab, ptr_dist)

        return final_vocab_dist, dec_hidden, enc_ctx_vector, dec_ctx_vector, enc_temporal_score
