import csv
import collections

from main.common.util.file_util import FileUtil
from main.common.vocab import *


class SimpleVocab(Vocab):

    def __init__(self, vocab_file, vocab_size=None):
        super(SimpleVocab, self).__init__({}, {})

        # read the vocab file

        vocab_map = {}

        with open(FileUtil.get_file_path(vocab_file), 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()

                if len(pieces) != 2:
                    logger.error('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue

                token = pieces[0]
                token_count = int(pieces[1])

                if token in [TK_PADDING['word'], TK_UNKNOWN['word'], TK_START_DECODING['word'], TK_STOP_DECODING['word']]:
                    raise Exception('[UNK], [PAD], [START] and [STOP] should not be in the vocab file, but %s is' % token)

                vocab_map[token] = token_count

        vocab_counter = collections.Counter(vocab_map)

        # build vocab

        count = 0  # keeps track of total number of words in the Vocab

        # [PAD], [UNK], [START] and [STOP]
        for token in [TK_PADDING, TK_UNKNOWN, TK_START_DECODING, TK_STOP_DECODING]:
            self._word2id[token['word']] = count
            self._id2word[count] = token['word']
            count += 1

        for token, nb in vocab_counter.most_common(vocab_size if vocab_size != -1 else None):
            self._word2id[token] = count
            self._id2word[count] = token
            count += 1

    def write_metadata(self, file_path):
        print("Writing word embedding metadata to %s..." % file_path)
        with open(file_path, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in range(self.size()):
                word = self._word2id[i]

                if word in [TK_PADDING['word'], TK_UNKNOWN['word'], TK_START_DECODING['word'], TK_STOP_DECODING['word']]:
                    continue

                writer.writerow({"word": word})
