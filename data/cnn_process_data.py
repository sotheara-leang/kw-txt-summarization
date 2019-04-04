import argparse
import os
import collections
import spacy

ptb_unescape = {'-lrb-': '(', '-rrb-': ')', '-lcb-': '{', '-rcb-': '}'}


def count_samples(file_in):
    counter = 0
    with open(file_in, 'r') as reader:
        while reader.readline() != '':
            counter += 1

    return counter


def generate_vocab(file_in, dir_out, max_vocab):
    nlp = spacy.load("en_core_web_sm")

    if not os.path.exists(dir_out):
        os.makedirs('extract')

    with open(file_in, 'r') as reader:
        vocab_counter = collections.Counter()

        # build vocab
        for article in reader:
            text = nlp(u"" + article)
            words = [token.text.lower() for token in text if token.is_space != True
                     and token.is_stop != True
                     and token.is_digit != True
                     and token.text not in ptb_unescape.keys()]

            vocab_counter.update(words)

            if len(vocab_counter) >= max_vocab:
                break

        # write vocab
        with open(dir_out + '/vocab2.bin', 'w') as writer:
            for token in vocab_counter:
                count = vocab_counter[token]

                writer.write(token + ' ' + str(count) + '\n')


def extract_samples(file_in, start_index, end_index, dir_out):
    path, filename = os.path.split(file_in)

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    counter = 0

    with open(file_in, 'r') as reader, open(dir_out + '/' + filename, 'w') as writer:
        while counter <= end_index:
            line = reader.readline()

            if line == '':
                break

            if counter < start_index:
                counter += 1
                continue

            line = line.strip()
            line = line[line.find('--') + 2:]
            for abbr, sign in ptb_unescape.items():
                line = line.replace(abbr, sign)

            if line == '':
                continue

            writer.write(line + '\n')

            counter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--opt', type=str, default="extract")
    parser.add_argument('--file', type=str)
    parser.add_argument('--dir_out', type=str, default="extract")
    parser.add_argument('--max_vocab', type=int, default="1000")
    parser.add_argument('--sindex', type=int, default="0")
    parser.add_argument('--eindex', type=int, default="999")

    args = parser.parse_args()

    if args.opt == 'chunk':
        pass
    elif args.opt == 'gen-vocab':
        generate_vocab(args.file, args.dir_out, args.max_vocab)
    elif args.opt == 'count':
        print(count_samples(args.file))
    else:
        extract_samples(args.file, args.sindex, args.eindex)
