import os
import argparse
import collections
import re


def count_samples(file_name):
    counter = 0
    with open(file_name, 'r') as reader:

        while reader.readline() != '':
            counter += 1

    return counter


def extract_samples(file_name, start_index, end_index, extract_file_name=None):
    if not extract_file_name:
        extract_file_name = file_name

    if not os.path.exists('extract'):
        os.makedirs('extract')

    counter = 0
    eof = False

    with open('raw/' + file_name, 'r') as reader, open('extract/' + extract_file_name, 'w') as writer:
        while counter <= end_index:
            line = reader.readline()

            if line == '':
                eof = True
                break

            if counter < start_index:
                counter += 1
                continue

            line = normalize_string(line)

            writer.write(line + '\n')

            counter += 1

    return eof


def chunk_samples(file_name, chunk_size):
    if not os.path.exists('extract'):
        os.makedirs('extract')

    with open('raw/' + file_name, 'r') as reader:
        counter = 0
        sindex = 0
        eindex = chunk_size

        while True:
            extract_file_name = file_name + '_' + str(counter + 1)

            eof = extract_samples(file_name, sindex, eindex, extract_file_name)
            if eof is True:
                break

            sindex = eindex + 1
            eindex = sindex + chunk_size
            counter += 1


def generate_vocab(file, vocab_file=None):
    if not os.path.exists('extract'):
        os.makedirs('extract')

    with open(file, 'r') as reader:
        vocab_counter = collections.Counter()

        # build vocab
        for article in reader:
            tokens = normalize_string(article).split(' ')

            tokens = [t.strip() for t in tokens]  # strip
            tokens = [t for t in tokens if valid_vocab(t)]  # remove invalid

            vocab_counter.update(tokens)

        # write vocab
        if vocab_file is None:
            vocab_file = 'extract/vocab.txt'

        with open(vocab_file, 'w') as writer:
            for token in vocab_counter:
                count = vocab_counter[token]
                writer.write(token + ' ' + str(count) + '\n')


def normalize_string(string):
    return string.lower().strip()


def valid_vocab(string):
    return string != '' and re.match('^([a-z]+)|([a-z]+-[a-z]+)|[\'\"(),.?!]$', string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--opt', type=str, default="extract")
    parser.add_argument('--file', type=str)
    parser.add_argument('--sindex', type=int, default="0")
    parser.add_argument('--eindex', type=int, default="999")
    parser.add_argument('--chunk_size', type=int, default="20000")
    parser.add_argument('--vocab_file', type=str)

    args = parser.parse_args()

    if args.opt == 'chunk':
        chunk_samples(args.file, args.chunk_size)
    elif args.opt == 'gen-vocab':
        generate_vocab(args.file, args.vocab_file)
    elif args.opt == 'count':
        print(count_samples(args.file))
    else:
        extract_samples(args.file, args.sindex, args.eindex)
