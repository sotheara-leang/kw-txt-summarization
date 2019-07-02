import argparse
import pickle
import os
import numpy as np
import tqdm


def count(file_in):
    counter = 0
    with open(file_in, 'r', encoding='utf-8') as f:
        for _ in f:
            counter += 1
    return counter


def generate_embedding(file, dir_out, fname):
    word2vect = {}

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    with open(file, 'r', encoding='utf-8') as f:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # gen_emb|count
    parser.add_argument('--opt', type=str, default='gen_emb')

    # glove embedding file
    parser.add_argument('--file', type=str)

    # output directory
    parser.add_argument('--dir_out', type=str, default='extract')

    # output file name
    parser.add_argument('--fname', type=str)

    args = parser.parse_args()

    opt = args.opt

    if opt == 'count':
        nb = count(args.file)
        print(nb)
    elif opt == 'gen_emb':
        generate_embedding(args.file, args.dir_out, args.fname)
