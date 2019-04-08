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


def extract_samples(file_in, start_index, end_index, dir_out):
    path, filename = os.path.split(file_in)

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    counter = 0
    eof = False

    with open(file_in, 'r') as reader, open(dir_out + '/' + filename, 'w') as writer:
        while counter <= end_index:
            line = reader.readline()

            if line == '':
                eof = True
                break

            if counter < start_index:
                counter += 1
                continue

            line = normalize_string(line)
            line = line.strip()

            if line == '':
                continue

            writer.write(line + '\n')

            counter += 1

    return eof


def generate_vocab(files_in, dir_out, vocab_fname, max_vocab):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    reach_max_vocab = False
    vocab_counter = collections.Counter()

    for file in files_in:
        with open(file, 'r') as reader:

            # build vocab
            for line in reader:

                if line == '':
                    break

                tokens = normalize_string(line).split(' ')

                valid_tokens = []
                for token in tokens:
                    token = token.strip()
                    if not valid_vocab(token):
                        continue

                    valid_tokens.append(token)

                vocab_counter.update(valid_tokens)

                if max_vocab > 0 and len(vocab_counter) >= max_vocab:
                    reach_max_vocab = True
                    break

        if reach_max_vocab is True:
            break

    output_fname = 'vocab.txt' if vocab_fname is None else vocab_fname

    with open(dir_out + '/' + output_fname, 'w') as writer:
        for i, token in enumerate(vocab_counter):
            if max_vocab > 0 and i >= max_vocab:
                break

            count = vocab_counter[token]
            writer.write(token + ' ' + str(count) + '\n')


def normalize_string(string):
    return string.lower()


def valid_vocab(string):
    return string != '' and re.match('^([a-z]+)|[a-z]+(-[a-z]+)+|[\'\"(),.?!]|\'(a-z)+$', string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--opt', type=str, default="gen-vocab")
    parser.add_argument('--file', '--names-list', nargs="*")
    parser.add_argument('--dir_out', type=str, default="extract")
    parser.add_argument('--max_vocab', type=int, default="-1")
    parser.add_argument('--sindex', type=int, default="0")
    parser.add_argument('--eindex', type=int, default="999")
    parser.add_argument('--vocab_fname', type=str, default="vocab.txt")

    args = parser.parse_args()

    if args.opt == 'gen-vocab':
        generate_vocab(args.file, args.dir_out, args.vocab_fname, args.max_vocab)
    elif args.opt == 'count':
        print(count_samples(args.file))
    else:
        extract_samples(args.file[0], args.sindex, args.eindex, args.dir_out)
