import argparse
import numpy as np
import pickle
import os
import bcolz


def generate_vocab(file_in, dir_out):
    word2id = {}
    id2word = {}
    id2vect = {}

    vectors = bcolz.carray(np.zeros(1), rootdir=dir_out + '/embedding', mode='w')

    idx = 4

    with open(file_in, 'r') as f:
        for line in f:
            line = line.split()

            word = line[0]
            vector = np.array(line[1:]).astype(np.float)

            word2id[word] = idx
            id2word[idx] = word
            id2vect[idx] = vector

            vectors.append(vector)

            idx += 1

    if not os.path.exists(dir_out):
        os.makedirs(dir_out + '/embedding')

    vocab = {
        'word2id': word2id,
        'id2word': id2word
    }

    pickle.dump(vocab, open(dir_out + '/vocab.bin', 'wb'))

    vectors = bcolz.carray(vectors[1:].reshape(-1, vector.shape[0]), rootdir=dir_out + '/embedding', mode='w')
    vectors.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in', type=str)
    parser.add_argument('--dir_out', type=str, default='extract')

    args = parser.parse_args()

    generate_vocab(args.file_in, args.dir_out)
