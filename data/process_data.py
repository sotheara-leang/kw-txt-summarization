import os
import argparse


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--opt', type=str, default="extract")
    parser.add_argument('--file', type=str)
    parser.add_argument('--sindex', type=int, default="0")
    parser.add_argument('--eindex', type=int, default="1000")
    parser.add_argument('--chunk_size', type=int, default="20000")

    args = parser.parse_args()

    file = args.file

    if args.opt == 'chunk':
        chunk_samples(file, args.chunk_size)
    else:
        extract_samples(file, args.sindex, args.eindex)

