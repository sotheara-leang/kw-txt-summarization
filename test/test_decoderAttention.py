from unittest import TestCase

import torch as t

from main.common.common import *
from main.decoder_attention import DecoderAttention


class TestDecoderAttention(TestCase):

    def test(self):
        dec_hidden = t.randn(2, 2 * conf.get('hidden-size'))
        pre_dec_hidden = t.randn(2, 3, 2 * conf.get('hidden-size'))

        decoder_att = DecoderAttention()

        context_vector = decoder_att(dec_hidden, pre_dec_hidden)

        print(context_vector, context_vector.size())
