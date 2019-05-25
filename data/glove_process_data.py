import argparse
import pickle

import numpy as np
import tqdm


def count(file_in):
    counter = 0
    with open(file_in, 'r', encoding='utf-8') as f:
        for line in f:
            counter += 1
    return counter


def generate_embedding(file_in, dir_out, fname):
    word2vect = {}

    with open(file_in, 'r', encoding='utf-8') as f:
        for line in tqdm.tqdm(f):
            line = line.split()

            word = line[0]
            embedding = np.array(line[1:]).astype(np.float)

            word2vect[word] = embedding

    data = {
        'word2vect': word2vect,
        'vocab': list(word2vect.keys())
    }

    output_fname = 'embedding.bin' if fname is None else fname

    with open(dir_out + '/' + output_fname, 'wb') as f:
        pickle.dump(data, f)


def generate_vocab(file_in, dir_out, fname):
    output_fname = 'glove-vocab.txt' if fname is None else fname

    with open(file_in, 'r') as r, open(dir_out + '/' + output_fname, 'w') as w:
        words = []
        for line in tqdm.tqdm(r):
            line = line.split()
            word = line[0]

            words.append(word)

        for word in words:
            w.write(word + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='gen_vocab')
    parser.add_argument('--file_in', type=str)
    parser.add_argument('--dir_out', type=str, default='extract')
    parser.add_argument('--fname', type=str)

    args = parser.parse_args()

    opt = args.opt

    if opt == 'count':
        nb = count(args.file_in)
        print(nb)
    elif opt == 'gen_vocab':
        generate_vocab(args.file_in, args.dir_out, args.fname)
    elif opt == 'gen_emb':
        generate_embedding(args.file_in, args.dir_out, args.fname)
