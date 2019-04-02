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

    def words2ids(self, words):
        return [self.word2id(w) for w in words]

    def extend_words2ids(self, words, oovs=[]):
        ids = []
        oovs_ = oovs
        for w in words:
            i = self.word2id(w)
            if i == TK_UNKNOWN.idx:  # If w is OOV
                if w in oovs:  # If w is in OOV
                    vocab_idx = self.size() + oovs.index(w)  # Map to its temporary article OOV number
                    ids.append(vocab_idx)
                else:  # If w is an out-of-article OOV
                    ids.append(TK_UNKNOWN.idx)  # Map to the UNK token id
            else:
                ids.append(i)
        return ids, oovs_

    def ids2words(self, ids, oovs=None):
        words = []
        for i in ids:
            try:
                w = self.id2word(i)  # might be [UNK]
            except ValueError:  # w is OOV
                if oovs is not None:
                    oov_idx = i - self.size()
                    try:
                        w = oovs[oov_idx]
                    except ValueError:
                        raise ValueError('Error: a word ID %i does not corresponds to OOV %i' % (i, oov_idx))
                else:
                    w = TK_UNKNOWN.word
            words.append(w)
        return words

    def show_oovs(self, summary, oovs: None):
        words = summary.split(' ')
        new_words = []

        for w in words:
            if self.word2id(w) == TK_UNKNOWN.idx:  # w is oov
                if oovs is None:  # baseline mode
                    new_words.append("__%s__" % w)
                else:  # pointer-generator mode
                    if w in oovs:
                        new_words.append("__%s__" % w)
                    else:
                        new_words.append("!!__%s__!!" % w)
            else:  # w is in-vocab word
                new_words.append(w)

        out_str = ' '.join(new_words)
        return out_str

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
