import argparse
import collections
import os

import tqdm

escape = {'#S#': ' '}


def generate_vocab(files, dir_out, fname, max_vocab):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    reach_max_vocab = False
    vocab_counter = collections.Counter()

    for file in files:
        with open(file, 'r', encoding='utf-8') as reader:

            for line in tqdm.tqdm(reader):

                for abbr, sign in escape.items():
                    line = line.replace(abbr, sign)

                if line == '':
                    break

                for abbr, sign in escape.items():
                    line = line.replace(abbr, sign)

                tokens = line.split()

                tokens = [token.strip() for token in tokens if token.strip() != '']

                vocab_counter.update(tokens)

                if max_vocab > 0 and len(vocab_counter) >= max_vocab:
                    reach_max_vocab = True
                    break

        if reach_max_vocab is True:
            break

    with open(dir_out + '/' + fname, 'w', encoding='utf-8') as writer:
        vocab_counter = sorted(vocab_counter.items(), key=lambda e: e[1], reverse=True)

        for i, element in enumerate(vocab_counter):
            if max_vocab > 0 and i >= max_vocab:
                break

            token = element[0]
            count = element[1]

            writer.write(token + ' ' + str(count) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # article file, summary file
    parser.add_argument('--files', nargs="*")

    # output file name
    parser.add_argument('--fname', type=str, default="vocab.txt")

    # vocabulary size
    parser.add_argument('--max_vocab', type=int, default="-1")

    # output dir
    parser.add_argument('--dir_out', type=str, default="extract")

    args = parser.parse_args()

    generate_vocab(args.files, args.dir_out, args.fname, args.max_vocab)
