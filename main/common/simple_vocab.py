import collections
import csv

from main.common.vocab import *


class SimpleVocab(Vocab):

    def __init__(self, vocab_file, vocab_size=None):
        super(SimpleVocab, self).__init__({}, {})

        self.logger = logger(self)

        # read the vocab file

        vocab_map = {}

        if not os.path.isfile(vocab_file):
            raise Exception('vocab file not exist: %s' % vocab_file)

        self.logger.debug('initialize vocabulary from: %s', vocab_file)

        with open(vocab_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                pieces = line.split()

                if len(pieces) != 2:
                    self.logger.warning('incorrectly formatted line in vocabulary file: %s', line)
                    continue

                token = pieces[0]
                token_count = int(pieces[1])

                if token in [TK_PADDING['word'], TK_UNKNOWN['word'], TK_START['word'], TK_STOP['word']]:
                    raise Exception('[UNK], [PAD], [START] and [STOP] should not be in the vocab file, but %s is' % token)

                vocab_map[token] = token_count

        vocab_counter = collections.Counter(vocab_map)

        # build vocab

        count = 0  # keeps track of total number of words in the Vocab

        # [PAD], [UNK], [START] and [STOP]
        for token in [TK_PADDING, TK_UNKNOWN, TK_START, TK_STOP]:
            self._word2id[token['word']] = count
            self._id2word[count] = token['word']
            count += 1

        for token, nb in vocab_counter.most_common(vocab_size - 4 if vocab_size != -1 else None):   # not include 4 predefined tokens
            self._word2id[token] = count
            self._id2word[count] = token
            count += 1

    def write_metadata(self, file_path):
        self.logger.debug("Writing word embedding metadata to %s...", file_path)
        with open(file_path, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in range(self.size()):
                word = self._word2id[i]

                if word in [TK_PADDING['word'], TK_UNKNOWN['word'], TK_START['word'], TK_STOP['word']]:
                    continue

                writer.writerow({"word": word})
