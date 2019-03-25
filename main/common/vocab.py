# Most codes are from https://github.com/rohithreddy024/Text-Summarizer-Pytorch/blob/master/data_util/data.py

import csv

from main.common.common import *
from main.common.util.file_util import FileUtil


class Token(object):

    def __init__(self, word, idx=-1):
        self.word = word
        self.idx = idx


TK_START_SENTENCE   = Token('<s>')
TK_END_SENTENCE     = Token('</s>')

TK_PADDING          = Token('[PAD]',    0)
TK_UNKNOWN          = Token('[UNK]',    1)
TK_START_DECODING   = Token('[START]',  2)
TK_STOP_DECODING    = Token('[STOP]',   3)


class Vocab(object):

    def __init__(self, vocab_file):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab

        # [PAD], [UNK], [START] and [STOP]
        for token in [TK_PADDING, TK_UNKNOWN, TK_START_DECODING, TK_STOP_DECODING]:
            self._word_to_id[token.idx] = self._count
            self._id_to_word[self._count] = token.word
            self._count += 1

        # Read the vocab file
        with open(FileUtil.get_file_path(vocab_file), 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()

                if len(pieces) != 2:
                    logger.error('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue

                token = pieces[0]
                if token in [TK_PADDING.word, TK_UNKNOWN.word, TK_START_DECODING.word, TK_STOP_DECODING.word]:
                    raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] should not be in the vocab file, but %s is' % token)

                if token in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % token)

                self._word_to_id[token] = self._count
                self._id_to_word[self._count] = token
                self._count += 1

    def word2id(self, word):
        if word not in self._word_to_id:
            return TK_UNKNOWN.idx
        return self._word_to_id[word]

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        return self._count

    def write_metadata(self, file_path):
        print("Writing word embedding metadata to %s..." % file_path)
        with open(file_path, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in range(self.size()):
                word = self._id_to_word[i]

                if word in [TK_PADDING.word, TK_UNKNOWN.word, TK_START_DECODING.word, TK_STOP_DECODING.word]:
                    continue

                writer.writerow({"word": word})


def article2ids(article_words, vocab):
    ids = []
    oovs = []
    unk_id = TK_UNKNOWN.idx
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w)  # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num)  # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def summary2ids(summary_words, vocab, article_oovs):
    ids = []
    unk_id = TK_UNKNOWN.idx
    for w in summary_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w)  # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words


def show_art_oovs(article, vocab):
    unk_token = TK_UNKNOWN.idx
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w) == unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
    unk_token = TK_UNKNOWN.idx
    words = abstract.split(' ')
    new_words = []
    for w in words:
        if vocab.word2id(w) == unk_token:  # w is oov
            if article_oovs is None:  # baseline mode
                new_words.append("__%s__" % w)
            else:  # pointer-generator mode
                if w in article_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else:  # w is in-vocab word
            new_words.append(w)
    out_str = ' '.join(new_words)
    return out_str
