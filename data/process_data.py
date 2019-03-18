import os
import argparse
import collections
import re

def count_samples(file_name):
    counter = 0
    with open('raw/' + file_name, 'r') as reader:

        while reader.readline() != '':
            counter += 1
            print(counter)

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

            line = line.replace('``', '\"').replace('\'\'', '\"').replace('<t>', '').replace('</t>', '')

            if line != '\n':
                writer.write(line)

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


def generate_vocab(article_file, summary_file, vocab_file=None):
    if not os.path.exists('extract'):
        os.makedirs('extract')

    with open(article_file, 'r') as art_reader, open(summary_file, 'r') as summary_reader:
        vocab_counter = collections.Counter()

        # build vocab
        for article in art_reader:
            summary = next(summary_reader)

            art_tokens = article.split(' ')
            sum_tokens = summary.split(' ')

            tokens = art_tokens + sum_tokens
            tokens = [t.strip() for t in tokens]  # strip
            tokens = [t for t in tokens if t != ""]  # remove empty

            vocab_counter.update(tokens)

        # write vocab
        if vocab_file is None:
            vocab_file = 'extract/vocab.txt'

        with open(vocab_file, 'w') as writer:
            num_rex = re.compile('#+.?#*')

            for word in vocab_counter:
                if num_rex.match(word) is not None:
                    continue

                count = vocab_counter[word]
                writer.write(word + ' ' + str(count) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--opt', type=str, default="extract")
    parser.add_argument('--file', type=str)
    parser.add_argument('--sindex', type=int, default="0")
    parser.add_argument('--eindex', type=int, default="1000")
    parser.add_argument('--chunk_size', type=int, default="20000")

    # generate vocab
    parser.add_argument('--art_file', type=str)
    parser.add_argument('--sum_file', type=str)
    parser.add_argument('--vocab_file', type=str)

    args = parser.parse_args()

    if args.opt == 'chunk':
        chunk_samples(args.file, args.chunk_size)
    elif args.opt == 'gen-vocab':
        generate_vocab(args.art_file, args.sum_file, args.vocab_file)
    else:
        extract_samples(args.file)

