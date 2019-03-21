from unittest import TestCase

import torch as t

from main.common.common import *
from main.encoder import Encoder


class TestEncoder(TestCase):

    def test(self):
        x = t.randn(2, 3, conf.get('hidden-size'))
        seq_len = t.FloatTensor([3, conf.get('hidden-size')])

        encoder = Encoder()

        output, (h_n, c_n) = encoder(x, seq_len)

        print(output, output.size())
        print(h_n, h_n.size())
        print(c_n, c_n.size())

