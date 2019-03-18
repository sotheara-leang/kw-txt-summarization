# Most codes are from https://github.com/rohithreddy024/Text-Summarizer-Pytorch/blob/master/data_util/batcher.py

import numpy as np

from main.common.common import *
from main.common.vocab import *


class Batch(object):

    def __init__(self, sample_list, vocab, batch_size):
        self.batch_size = batch_size
        self.pad_id = vocab.word2id(PAD_TOKEN)

        self.init_encoder_seq(sample_list)      # initialize the input to the encoder
        self.init_decoder_seq(sample_list)      # initialize the input and targets for the decoder

    def init_encoder_seq(self, sample_list):
        # Determine the maximum length of the encoder input sequence in this batch
        max_enc_seq_len = max([ex.enc_len for ex in sample_list])

        # Pad the encoder input sequences up to the length of the longest sequence
        for sample in sample_list:
            sample.pad_encoder_input(max_enc_seq_len, self.pad_id)

        # Initialize the numpy arrays
        # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
        self.enc_batch = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        self.enc_lens = np.zeros((self.batch_size), dtype=np.int32)
        self.enc_padding_mask = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.float32)

        # Fill in the numpy arrays
        for i, sample in enumerate(sample_list):
            self.enc_batch[i, :] = sample.enc_input[:]
            self.enc_lens[i] = sample.enc_len
            for j in range(sample.enc_len):
                self.enc_padding_mask[i][j] = 1

        # For pointer-generator mode, need to store some extra info
        # Determine the max number of in-article OOVs in this batch
        self.max_art_oovs = max([len(sample.article_oovs) for sample in sample_list])

        # Store the in-article OOVs themselves
        self.art_oovs = [sample.article_oovs for sample in sample_list]

        # Store the version of the enc_batch that uses the article OOV ids
        self.enc_batch_extend_vocab = np.zeros((self.batch_size, max_enc_seq_len), dtype=np.int32)
        for i, sample in enumerate(sample_list):
            self.enc_batch_extend_vocab[i, :] = sample.enc_input_extend_vocab[:]

    def init_decoder_seq(self, sample_list):
        # Pad the inputs and targets
        for sample in sample_list:
            sample.pad_decoder_inp_targ(conf.get('max-dec-steps'), self.pad_id)

        # Initialize the numpy arrays.
        self.dec_batch = np.zeros((self.batch_size, conf.get('max-dec-steps')), dtype=np.int32)
        self.target_batch = np.zeros((self.batch_size, conf.get('max-dec-steps')), dtype=np.int32)

        # self.dec_padding_mask = np.zeros((self.batch_size, config.max_dec_steps), dtype=np.float32)
        self.dec_lens = np.zeros(self.batch_size, dtype=np.int32)

        # Fill in the numpy arrays
        for i, sample in enumerate(sample_list):
            self.dec_batch[i, :] = sample.dec_input[:]
            self.target_batch[i, :] = sample.target[:]
            self.dec_lens[i] = sample.dec_len
            # for j in range(sample.dec_len):
            #   self.dec_padding_mask[i][j] = 1

