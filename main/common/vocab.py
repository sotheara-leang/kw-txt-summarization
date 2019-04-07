from main.common.common import *


TK_PADDING          = {'word': '[PAD]',     'id': 0}
TK_UNKNOWN          = {'word': '[UNK]',     'id': 1}
TK_START_DECODING   = {'word': '[START]',   'id': 2}
TK_STOP_DECODING    = {'word': '[STOP]',    'id': 3}


class Vocab(object):

    def __init__(self, word2id, id2word):
        self._word2id = word2id
        self._id2word = id2word

    def word2id(self, word):
        if word not in self._word2id:
            return None
        return self._word2id[word]

    def id2word(self, word_id):
        if word_id not in self._id2word:
            return None
        return self._id2word[word_id]

    def size(self):
        return len(self._word2id)

    def id_exists(self, word_id):
        return False if self.id2word(word_id) is None else True

    def word_exists(self, word):
        return False if self.word2id(word) is None else True

    def words2ids(self, words, oovs=None):
        ids_ = []
        for w in words:
            id_ = self.word2id(w)
            if id_ is None:
                if oovs is not None and len(oovs) > 0:
                    try:
                        id_ = self.size() + oovs.index(w)
                    except ValueError:
                        #logger.warning('word ￿"%s￿" not found in %s', w, oovs)
                        id_ = TK_UNKNOWN['id']
                else:
                    id_ = TK_UNKNOWN['id']

            ids_.append(id_)
        return ids_

    def ids2words(self, ids, oovs=None):
        words = []
        for id_ in ids:
            w = self.id2word(id_)
            if w is None:
                if oovs is not None and len(oovs) > 0:
                    try:
                        w = oovs[id_ - self.size()]
                    except IndexError:
                        logger.warning('word id ￿"%s" not found in %s' + str(id_), oovs)
                        w = TK_UNKNOWN['word']
                else:
                    w = TK_UNKNOWN['word']

            words.append(w)
        return words

    def extend_words2ids(self, words):
        ids = []
        oovs = []
        for w in words:
            id_ = self.word2id(w)
            if id_ is None:
                if w not in oovs:
                    oovs.append(w)

                id_ = self.size() + oovs.index(w)

            ids.append(id_)
        return ids, oovs

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
