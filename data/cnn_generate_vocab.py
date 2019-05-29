import argparse
import collections
import os
import tqdm

escape = {'#S#': ' ', '#Q#': ' '}


''' 
    opt:
        vocab
            file_in: article, summary file
            
        entity-vocab
            file_in: keyword file
'''
def generate_vocab(files_in, dir_out, fname, max_vocab, option):
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    reach_max_vocab = False
    vocab_counter = collections.Counter()

    for file in files_in:
        with open(file, 'r', encoding='utf-8') as reader:

            # build vocab
            for line in tqdm.tqdm(reader):

                for abbr, sign in escape.items():
                    line = line.replace(abbr, sign)

                if line == '':
                    break

                if option == 'entity':
                    line = line.replace(',', ' ')

                tokens = line.split()

                tokens = [token.strip() for token in tokens if token.strip() != '']

                vocab_counter.update(tokens)

                if max_vocab > 0 and len(vocab_counter) >= max_vocab:
                    reach_max_vocab = True
                    break

        if reach_max_vocab is True:
            break

    if option == 'vocab':
        output_fname = 'vocab.txt' if fname is None else fname
    elif option == 'entity':
        output_fname = 'entity-vocab.txt' if fname is None else fname

    with open(dir_out + '/' + output_fname, 'w', encoding='utf-8') as writer:
        vocab_counter = sorted(vocab_counter.items(), key=lambda e: e[1], reverse=True)

        for i, element in enumerate(vocab_counter):
            if max_vocab > 0 and i >= max_vocab:
                break

            token = element[0]
            count = element[1]

            writer.write(token + ' ' + str(count) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--opt', type=str, default="vocab")   # vocab|entity
    parser.add_argument('--file', nargs="*")
    parser.add_argument('--fname', nargs="*")
    parser.add_argument('--max_vocab', type=int, default="-1")
    parser.add_argument('--dir_out', type=str, default="extract")

    args = parser.parse_args()

    generate_vocab(args.file, args.dir_out, args.fname[0] if args.fname is not None else None, args.max_vocab, args.opt)

