import argparse
import os
from random import shuffle
import tqdm


def extract_samples(files, number, dir_out, fnames):
    counter_ = 1

    samples = []

    with open(files[0], 'r', encoding='utf-8') as art_reader, \
            open(files[1], 'r', encoding='utf-8') as key_reader, \
            open(files[2], 'r', encoding='utf-8') as sum_reader:

        for article in tqdm.tqdm(art_reader):

            if number > 0 and counter_ >= number + 1:
                break

            summary = next(sum_reader)

            if article == '' or summary == '':
                continue

            keyword = next(key_reader)

            entities = keyword.strip().split(',')
            counter_ += len(entities)

            samples.append((article, keyword, summary))

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    _, art_fname = os.path.split(files[0])
    _, key_fname = os.path.split(files[1])
    _, sum_fname = os.path.split(files[2])

    art_output_fname = art_fname if fnames is None else fnames[0]
    key_output_fname = key_fname if fnames is None else fnames[1]
    sum_output_fname = sum_fname if fnames is None else fnames[2]

    shuffle(samples)

    with open(dir_out + '/' + art_output_fname, 'w', encoding='utf-8') as art_writer, \
            open(dir_out + '/' + key_output_fname, 'w', encoding='utf-8') as key_writer, \
            open(dir_out + '/' + sum_output_fname, 'w', encoding='utf-8') as sum_writer:
        for sample in samples:
            art_writer.write(sample[0])
            key_writer.write(sample[1])
            sum_writer.write(sample[2])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # number of examples to be extracted
    parser.add_argument('--num', type=int, default=-1)

    #  article file, keyword file, summary file
    parser.add_argument('--file', nargs="*")

    # article file, keyword file, summary file
    parser.add_argument('--fname', nargs="*")

    # output dir
    parser.add_argument('--dir_out', type=str, default="extract")

    option = parser.parse_args()

    extract_samples(option.file, option.num, option.dir_out, option.fname)
