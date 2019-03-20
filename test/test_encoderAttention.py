from unittest import TestCase

import torch as t

from main.common.common import *
from main.encoder_attention import EncoderAttention


class TestEncoderAttention(TestCase):

    def test(self):
        dec_hidden = t.randn(2, 2 * conf.get('hidden-size'))

        enc_hidden = t.randn(2, 3, 2 * conf.get('hidden-size'))

        sum_score = t.zeros(2, 3).fill_(1e-10)

        encoder_attention = EncoderAttention()

        context_vector, attention, sum_score = encoder_attention(dec_hidden, enc_hidden, sum_score)

        print(context_vector, context_vector.size())
        print(sum_score, sum_score.size())
