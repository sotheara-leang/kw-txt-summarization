import argparse
import os

import tqdm

'''
    file_in: keyword file
'''
def count_example(file_in):
    counter_ = 0
    with open(file_in, 'r', encoding='utf-8') as reader:
        for line in tqdm.tqdm(reader):
            entities = line.split(',')
            counter_ += len(entities)

    return counter_

def extract_samples(file_in, number, dir_out, fname):
    counter_ = 1

    samples = []

    with open(file_in[0], 'r', encoding='utf-8') as art_reader, \
            open(file_in[1], 'r', encoding='utf-8') as key_reader, \
            open(file_in[2], 'r', encoding='utf-8') as sum_reader:

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

    _, art_fname = os.path.split(file_in[0])
    _, key_fname = os.path.split(file_in[1])
    _, sum_fname = os.path.split(file_in[2])

    art_output_fname = art_fname if fname is None else fname[0]
    key_output_fname = key_fname if fname is None else fname[1]
    sum_output_fname = sum_fname if fname is None else fname[2]

    with open(dir_out + '/' + art_output_fname, 'w', encoding='utf-8') as art_writer, \
            open(dir_out + '/' + key_output_fname, 'w', encoding='utf-8') as key_writer, \
            open(dir_out + '/' + sum_output_fname, 'w', encoding='utf-8') as sum_writer:
        for sample in samples:
            art_writer.write(sample[0])
            key_writer.write(sample[1])
            sum_writer.write(sample[2])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--opt', type=str, default="extract")
    parser.add_argument('--num', type=int, default=-1)
    parser.add_argument('--file', nargs="*")
    parser.add_argument('--fname', nargs="*")
    parser.add_argument('--dir_out', type=str, default="extract")

    option = parser.parse_args()

    if option.opt == 'count':
        counter = count_example(option.file[0])
        print(counter)
    elif option.opt == 'extract':
        extract_samples(option.file, option.num, option.dir_out, option.fname)
