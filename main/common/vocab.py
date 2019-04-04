from main.common.common import *


TK_PADDING          = {'word': '[PAD]',     'id': 0}
TK_UNKNOWN          = {'word': '[UNK]',     'id': 1}
TK_START_DECODING   = {'word': '[START]',   'id': 2}
TK_STOP_DECODING    = {'word': '[STOP]',    'id': 3}


class Vocab(object):

    def __init__(self, word2id, id2word, id2vector):
        self._word2id = word2id
        self._id2word = id2word
        self._id2vector = id2vector

    def word2id(self, word):
        if word not in self._word2id:
            return TK_UNKNOWN['id']
        return self._word2id[word]

    def id2word(self, word_id):
        if word_id not in self._id2word:
            return TK_UNKNOWN['word']
        return self._id2word[word_id]

    def size(self):
        return len(self._word2id)

    def id_exists(self, word_id):
        return True if self.id2word(word_id) == TK_UNKNOWN['word'] else False

    def word_exists(self, word):
        return True if self.word2id(word) == TK_UNKNOWN['id'] else False

    def words2ids(self, words):
        return [self.word2id(w) for w in words]

    def ids2words(self, ids, oovs=None):
        words = []
        for id_ in ids:
            w = self.id2word(id_)
            if w is TK_UNKNOWN['word'] and oovs is not None:
                oov_idx = id_ - self.size()
                try:
                    w = oovs[oov_idx]
                except IndexError:
                    logger.warn('word id not found in oov: ' + str(id_))
                    w = TK_UNKNOWN['word']

            words.append(w)
        return words

    def extend_words2ids(self, words, oovs=[]):
        ids = []
        oovs_ = oovs
        for w in words:
            id_ = self.word2id(w)
            if id_ == TK_UNKNOWN['id'] and w in oovs:
                vocab_idx = self.size() + oovs.index(w)
                id_ = vocab_idx   
                
            ids.append(id_)
        return ids, oovs_

    def show_oovs(self, words, oovs: None):
        words = words.split(' ')
        new_words = []

        for w in words:
            if self.word2id(w) == TK_UNKNOWN['id']:  # w is oov
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
