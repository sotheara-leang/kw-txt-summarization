# Most of this file is copied form https://github.com/rohithreddy024/Text-Summarizer-Pytorch/blob/master/data_util/batcher.py

from main.common.common import *
from main.common.vocab import *


class Sample(object):

    def __init__(self, article, summary, vocab):
        self.article = article
        self.summary = summary

        # Get ids of special tokens
        start_decoding = vocab.word2id(START_DECODING)
        stop_decoding = vocab.word2id(STOP_DECODING)

        # Process the article
        article_words = article.split()
        if len(article_words) > conf.get('max-enc-steps'):
            article_words = article_words[:conf.get('max-enc-steps')]

        self.enc_len = len(article_words)  # store the length after truncation but before padding
        self.enc_input = [vocab.word2id(w) for w in
                          article_words]  # list of word ids; OOVs are represented by the id for UNK token

        # Process the summary
        summary_words = summary.split()  # list of strings
        abs_ids = [vocab.word2id(w) for w in
                   summary_words]  # list of word ids; OOVs are represented by the id for UNK token

        # Get the decoder input sequence and target sequence
        self.dec_input, _ = self.get_dec_inp_targ_seqs(abs_ids, conf.get('max-dec-steps'), start_decoding, stop_decoding)
        self.dec_len = len(self.dec_input)

        # If using pointer-generator mode, we need to store some extra info
        # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
        self.enc_input_extend_vocab, self.article_oovs = article2ids(article_words, vocab)

        # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
        abs_ids_extend_vocab = summary2ids(summary_words, vocab, self.article_oovs)

        # Get decoder target sequence
        _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, conf.get('max-dec-steps'), start_decoding, stop_decoding)

    # get decoder input target sequences
    def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
        inp = [start_id] + sequence[:]
        target = sequence[:]
        if len(inp) > max_len:  # truncate
            inp = inp[:max_len]
            target = target[:max_len]  # no end_token
        else:  # no truncation
            target.append(stop_id)  # end token
        assert len(inp) == len(target)
        return inp, target

    def pad_decoder_inp_targ(self, max_len, pad_id):
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)

    def pad_encoder_input(self, max_len, pad_id):
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        while len(self.enc_input_extend_vocab) < max_len:
            self.enc_input_extend_vocab.append(pad_id)
