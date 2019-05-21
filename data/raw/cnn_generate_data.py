# https://github.com/helmertz/querysum-data/blob/master/convert_rcdata.py

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import hashlib
import os
import re

import tqdm

import spacy
nlp = spacy.load("en")

SEP_QUERY           = ','
SEP_SUMMARY         = '#S#'
SEP_SUMMARY_QUERY   = '#Q#'
SEP_ENTITY          = ','

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'

END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"]  # acceptable ways to end a sentence


class Article:

    def __init__(self, article_text=None, query_to_summaries=None, entities=None):
        self.article_text = article_text
        self.query_to_summaries = {} if query_to_summaries is None else query_to_summaries
        self.entities = entities

class Question:
    def __init__(self, url=None, query=None, entities=None):
        self.url = url
        self.query = query
        self.entities = entities

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', nargs="*")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--max', type=int, default=-1, help='max questions to be processed')
    parser.add_argument('--extract', type=int, default=0, help='0: all data, 1: only valid data')

    options = parser.parse_args()

    options.input_dir = ['/home/vivien/Downloads/raw/cnn']
    options.output_dir = '/home/vivien/Downloads/generate'
    options.max = 50

    print('>>> extract datasets')
    datsets = extract_datasets(options)

    print('\n>>> write datasets')
    write_datasets(datsets, options)

def extract_datasets(options):
    datasets = {}

    for input_dir in options.input_dir:
        story_dir = path(input_dir, 'stories')

        question_dir = path(input_dir, 'questions')
        question_sub_dirs = os.listdir(question_dir)

        for question_sub_dir in question_sub_dirs:
            if os.path.isfile(path(question_dir, question_sub_dir)):
                continue

            print('\nextract questions from %s' % question_sub_dir)

            dataset = {}
            datasets[question_sub_dir] = dataset
            question_sub_dir = path(question_dir, question_sub_dir)
            question_counter = 0

            for question_file in tqdm.tqdm(os.listdir(question_sub_dir)):
                question_file = path(question_sub_dir, question_file)

                if not os.path.isfile(question_file):
                    continue

                if options.max > 0 and question_counter >= options.max:
                    break

                question = extract_question(question_file)

                if question is None or question.url in dataset:
                    continue

                article = extract_article(question, story_dir, options)

                if article is None:
                    continue

                dataset[question.url] = article

                question_counter += 1

    return datasets

def write_datasets(datasets, options):
    output_dir = options.output_dir

    for dataset_name, articles in datasets.items():

        print('write dataset %s' % dataset_name)

        # ignore articles where no query was found
        filtered_articles = [item for item in articles.items() if len(item[1].query_to_summaries) > 0]

        # sort article by text length
        sorted_articles = sorted(filtered_articles, key=lambda article_tuple: len(article_tuple[1].article_text), reverse=False)

        output_set_dir = path(output_dir, dataset_name)

        if not os.path.exists(output_set_dir):
            os.makedirs(output_set_dir)

        article_filename = dataset_name + '.article.txt'
        keyword_filename = dataset_name + '.keyword.txt'
        summary_filename = dataset_name + '.summary.txt'
        entity_filename = dataset_name + '.entity.txt'

        with open(os.path.join(output_set_dir, article_filename), 'w', encoding='utf-8') as article_file, \
                open(os.path.join(output_set_dir, summary_filename), 'w', encoding='utf-8') as summary_file, \
                open(os.path.join(output_set_dir, keyword_filename), 'w', encoding='utf-8') as keyword_file, \
                open(os.path.join(output_set_dir, entity_filename), 'w', encoding='utf-8') as entity_file:

            for url, article in tqdm.tqdm(sorted_articles):
                article_text = article.article_text
                query_to_summaries = article.query_to_summaries
                entities = article.entities

                # ignore if no queries were found for article
                if len(query_to_summaries) == 0:
                    continue

                # article
                article_file.write(article_text + '\n')

                # entity
                entity_file.write(SEP_ENTITY.join(entities) + '\n')

                # query and summaries
                query_to_summaries = sorted(query_to_summaries.items(), key=lambda query_tuple: hash_hex(query_tuple[0]))

                for idx, query_to_summary in enumerate(query_to_summaries):
                    query, summaries = query_to_summary

                    if idx > 0:
                        keyword_file.write(SEP_QUERY)
                        summary_file.write(SEP_SUMMARY_QUERY)

                    keyword_file.write(query)

                    # summary
                    for sum_idx, summary in enumerate(summaries):
                        if sum_idx > 0:
                            summary_file.write(SEP_SUMMARY)

                        summary_file.write(summary)

                keyword_file.write('\n')
                summary_file.write('\n')

def extract_question(question_file):
    lines = read_text_file(question_file)

    if len(lines) == 0:
        return None

    url = lines[0]
    placeholder = lines[6]
    entity_mapping_lines = lines[8:]

    entity_dictionary = get_entity_dictionary(entity_mapping_lines)

    query = entity_dictionary[placeholder].split()

    entities = [entity for entity in entity_dictionary.values()]

    return Question(url, query, entities)

def extract_article(question, story_dir, options):
    story_file = '{}.story'.format(path(story_dir, hash_hex(question.url)))

    if not os.path.isfile(story_file):
        return None

    lines = read_text_file(story_file)

    lines = [fix_missing_period(line) for line in lines]

    # put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)

    # separate out article and abstract sentences
    article_lines = []
    highlights = []

    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    if len(article_lines) == 0 or len(highlights) == 0:
        return

    query_highlights = []
    for highlight in highlights:
        if contains_sublist(highlight.split(), question.query):
            query_highlights.append(highlight)

    query_not_found = False
    if len(query_highlights) == 0:
        # For now, ignore if sequence of tokens not found in any highlight. It happens for example when query is
        # "American" and highlight contains "Asian-American".
        if options.extract == 1:
            return
        else:
            query_not_found = True
            query_highlights = highlights

    article = Article(' '.join(article_lines), None, question.entities)

    query = ' '.join(question.query) if query_not_found is False else ''
    article.query_to_summaries[query] = query_highlights

    return article

def read_text_file(text_file):
    with open(text_file, "r") as f:
        text = f.read().lower()

    text = ' '.join(tokenize(text))
    lines = text.splitlines()

    lines = [line.strip() for line in lines]

    return lines

def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line:
        return line
    if line == "" :
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + " ."

def get_entity_dictionary(entity_mapping_lines):
    entity_dictionary = {}
    for mapping in entity_mapping_lines:
        entity, name = mapping.split(':', 1)
        entity_dictionary[entity] = name
    return entity_dictionary

def tokenize(text):
    doc = nlp(u'' + text.lower())
    tokens = []
    for token in doc:
        tokens.append(token.text)
    return tokens

def path(path, *paths):
    return os.path.join(path, *paths)

def hash_hex(string):
    hash_ = hashlib.sha1()
    hash_.update(string.encode('utf-8'))
    return hash_.hexdigest()

def contains_sublist(list_, sublist):
    for i in range(len(list_)):
        if list_[i:(i + len(sublist))] == sublist:
            return True
    return False


if __name__ == '__main__':
    main()
