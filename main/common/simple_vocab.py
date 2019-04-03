import csv

from main.common.common import *
from main.common.util.file_util import FileUtil
from main.common.vocab import *


class SimpleVocab(Vocab):

    def __init__(self, vocab_file):
        super(SimpleVocab, self).__init__({}, {}, {})

        count = 0  # keeps track of total number of words in the Vocab

        # [PAD], [UNK], [START] and [STOP]
        for token in [TK_PADDING, TK_UNKNOWN, TK_START_DECODING, TK_STOP_DECODING]:
            self._word2id[token['id']] = count
            self._word2id[count] = token['word']
            count += 1

        # Read the vocab file
        with open(FileUtil.get_file_path(vocab_file), 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()

                if len(pieces) != 2:
                    logger.error('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue

                token = pieces[0]
                if token in [TK_PADDING['word'], TK_UNKNOWN['word'], TK_START_DECODING['word'], TK_STOP_DECODING['word']]:
                    raise Exception('[UNK], [PAD], [START] and [STOP] should not be in the vocab file, but %s is' % token)

                if token in self._word2id:
                    raise Exception('Duplicated word in vocabulary file: %s' % token)

                self._word2id[token] = count
                self._word2id[count] = token
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
