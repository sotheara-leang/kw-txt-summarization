# https://github.com/helmertz/querysum-data/blob/master/convert_rcdata.py, https://github.com/abisee/cnn-dailymail/blob/master/make_datafiles.py

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import hashlib
import os
import multiprocessing
import tqdm
import spacy
import math
import statistics

nlp = spacy.load("en")

SEP_SUMMARY = '#S#'
SEP_SUMMARY_QUERY = '#Q#'
SEP_ENTITY = '#E#'

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'

END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  # acceptable ways to end a sentence


class Story:

    def __init__(self, article_text=None, highlights=None, query_to_summaries=None, entities=None, f_name=None):
        self.article_text = article_text
        self.article_size = len(article_text.split())

        self.highlights = highlights
        self.query_to_summaries = {} if query_to_summaries is None else query_to_summaries
        self.entities = entities if entities is not None else []
        self.f_name = f_name
        self.query_mapping = {}


class Question:
    def __init__(self, url=None, query=None, entities=None, f_name=None):
        self.url = url
        self.query = query
        self.entities = entities
        self.f_name = f_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', type=str, default='generate', help='preprocess|generate')
    parser.add_argument('--input_dir', nargs="*")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--gen_all', type=int, default=1, help='1: all data, 0: only question/answer pair')
    parser.add_argument('--zip', type=int, default=1, help='1: zip data, 0: normal')
    parser.add_argument('--validation_test_fraction', type=float, default=0.025)

    options = parser.parse_args()

    if options.op == 'preprocess':
        print('>>> pre-process datafiles')
        preprocess_datafiles(options)

    elif options.op == 'generate':
        print('>>> process datafiles')
        datsets = extract_datasets(options)

        print('\n>>> write datafiles')
        write_datasets(datsets, options)


def preprocess_datafiles(options):
    pool = multiprocessing.Pool(int(os.cpu_count() / 2))

    for input_dir in options.input_dir:
        print('\npreprocess directory: %s' % input_dir)

        # story

        story_dir = path(input_dir, 'stories')

        print('\npreprocess story files in %s' % story_dir)

        story_files = os.listdir(story_dir)
        story_files = [path(story_dir, story_file) for story_file in story_files if story_file.endswith('.story')]

        list(tqdm.tqdm(pool.imap(tokenize_file, story_files), total=len(story_files)))


def extract_datasets(options):
    datasets = {}

    story_entities = extract_story_entities(options)

    total_counter = 0

    for input_dir in options.input_dir:
        print('\n>>> extract strories in %s' % input_dir)

        story_dir_path = path(input_dir, 'stories')

        dataset = {}
        example_counter = 0

        story_files = os.listdir(story_dir_path)
        for story_file in tqdm.tqdm(story_files):

            story = extract_story(path(story_dir_path, story_file))
            if story is None:
                continue

            dataset[story_file] = story

            if story_file not in story_entities:
                if options.gen_all == 1:
                    story.query_to_summaries[''] = story.highlights

                    example_counter += 1
                continue

            entities, question_file = story_entities.get(story_file)

            story.entities = entities
            highlights = [highlight.split() for highlight in story.highlights]

            for entity in entities:

                entity_highlights = []
                for highlight in highlights:
                    if contains_sublist(highlight, entity.split()):
                        entity_highlights.append(highlight)

                if len(entity_highlights) == 0:
                    continue

                story.query_to_summaries[entity] = [' '.join(highlight) for highlight in entity_highlights]
                story.query_mapping[entity] = question_file

                example_counter += 1

        total_counter += example_counter

        print('examples: ', example_counter)
        print('stories: ', len(dataset))

        datasets.update(dataset)

    print('\ntotal examples: ', total_counter)
    print('total stories: ', len(datasets))

    return datasets


def extract_story_entities(options):
    story_entities = {}

    for input_dir in options.input_dir:
        print('\n>>> extract entities in %s' % input_dir)

        question_dir_path = path(input_dir, 'questions')

        question_sub_dirs = os.listdir(question_dir_path)
        for question_sub_dir in question_sub_dirs:
            question_sub_dir_path = path(question_dir_path, question_sub_dir)

            print('\nextract entities in %s' % question_sub_dir_path)

            ds_story_entities = {}

            for question_file in tqdm.tqdm(os.listdir(question_sub_dir_path)):
                question_file = path(question_sub_dir_path, question_file)

                # extract question
                question = extract_question(question_file)
                if question is None:
                    continue

                story_file = '{}.story'.format(hash_hex(question.url))

                if story_file in ds_story_entities:
                    continue

                ds_story_entities[story_file] = (question.entities, question_file)

            print('element: %d' % len(ds_story_entities))

            story_entities.update(ds_story_entities)

    return story_entities


def display_datasets(dataset, options):
    article_len = []
    summary_len = []
    keyword_len = []
    example = 0
    doc_wo_question = 0

    for story_file, story in dataset:
        query_to_summaries = story.query_to_summaries

        if options.gen_all == 1 and '' in query_to_summaries:
            doc_wo_question += 1

        for query, summaries in query_to_summaries.items():
            keyword_len.append(len(query.split()))
            summary_len.append(len(' '.join(summaries).split()))
        example += len(query_to_summaries)

        article_len.append(story.article_size)

    print('examples: ', example)
    print('stories: ', len(dataset))
    print('stories-wo-question: ', doc_wo_question)
    print('max article len: ', max(article_len))
    print('max summary len: ', max(summary_len))
    print('max keyword len: ', max(keyword_len))
    print('avg article len: ', statistics.mean(article_len))
    print('avg summary len: ', statistics.mean(summary_len))
    print('avg keyword len: ', statistics.mean(keyword_len))


def write_datasets(datasets, options):
    output_dir = options.output_dir

    filtered_stories = [item for item in datasets.items()]
    if options.gen_all == 0:
        filtered_stories = [item for item in filtered_stories.items() if len(item[1].query_to_summaries) > 0]

    num_validation_test_ds = math.ceil(options.validation_test_fraction * len(filtered_stories))

    validation_stories = filtered_stories[:num_validation_test_ds]
    test_stories = filtered_stories[num_validation_test_ds:(2 * num_validation_test_ds)]
    training_stories = filtered_stories[(2 * num_validation_test_ds):]

    print('\nvalidation dataset: %d' % len(validation_stories))
    print('test dataset: %d' % len(test_stories))
    print('training dataset: %d' % len(training_stories))

    output_ds = [('validation', validation_stories),
                 ('test', test_stories),
                 ('training', training_stories)]

    for ds_name, ds_stories in output_ds:

        print('\n>>> write dataset: %s' % ds_name)

        display_datasets(ds_stories, options)

        continue

        article_filename = ds_name + '.article.txt'
        keyword_filename = ds_name + '.keyword.txt'
        summary_filename = ds_name + '.summary.txt'
        entity_filename = ds_name + '.entity.txt'
        mapping_filename = ds_name + '.mapping.txt'

        sorted_stories = sorted(ds_stories, key=lambda story_tuple: story_tuple[1].article_size, reverse=False)

        output_set_dir = path(output_dir, ds_name)
        if not os.path.exists(output_set_dir):
            os.makedirs(output_set_dir)

        with open(path(output_set_dir, article_filename), 'w', encoding='utf-8') as article_file, \
                open(path(output_set_dir, summary_filename), 'w', encoding='utf-8') as summary_file, \
                open(path(output_set_dir, keyword_filename), 'w', encoding='utf-8') as keyword_file, \
                open(path(output_set_dir, entity_filename), 'w', encoding='utf-8') as entity_file, \
                open(path(output_set_dir, mapping_filename), 'w', encoding='utf-8') as mapping_file:

            for story_file, story in tqdm.tqdm(sorted_stories):
                article_text = story.article_text
                query_to_summaries = story.query_to_summaries
                entities = story.entities

                if options.zip is 1:
                    article_file.write(article_text + '\n')
                    entity_file.write(SEP_ENTITY.join(entities) + '\n')

                    # mapping
                    mapping_file.write(story.f_name)
                    for _, mapping in story.query_mapping.items():
                        if options.zip == 1:
                            mapping_file.write(':%s' % mapping)
                        else:
                            mapping_file.write('%s\n' % mapping)

                    mapping_file.write('\n')

                for idx, query_to_summary in enumerate(query_to_summaries.items()):
                    query, summaries = query_to_summary

                    if options.zip == 0:
                        article_file.write(article_text + '\n')
                        keyword_file.write(query + '\n')
                        entity_file.write(SEP_ENTITY.join(entities) + '\n')

                        # summary
                        for sum_idx, summary in enumerate(summaries):
                            if sum_idx > 0:
                                summary_file.write(SEP_SUMMARY)

                            summary_file.write(summary)
                        summary_file.write('\n')

                        # mapping
                        for _, mapping in story.query_mapping.items():
                            mapping_file.write('%s:%s\n' % (story.f_name, mapping))

                    else:
                        if idx > 0:
                            keyword_file.write(SEP_ENTITY)
                            summary_file.write(SEP_SUMMARY_QUERY)

                        keyword_file.write(query)

                        # summary
                        for sum_idx, summary in enumerate(summaries):
                            if sum_idx > 0:
                                summary_file.write(SEP_SUMMARY)

                            summary_file.write(summary)

                if options.zip is 1:
                    keyword_file.write('\n')
                    summary_file.write('\n')


def extract_question(question_file):
    if not os.path.isfile(question_file) or not question_file.endswith('.question'):
        return None

    lines = read_text_file(question_file)

    if len(lines) == 0:
        return None

    url = lines[0]
    placeholder = lines[6]
    entity_mapping_lines = lines[8:]

    entity_dictionary = get_entity_dictionary(entity_mapping_lines)

    query = entity_dictionary[placeholder].split()

    entities = [entity for entity in entity_dictionary.values()]

    return Question(url, query, entities, question_file)


def get_entity_dictionary(entity_mapping_lines):
    entity_dictionary = {}
    for mapping in entity_mapping_lines:
        entity, name = mapping.split(':', 1)
        entity_dictionary[entity] = name
    return entity_dictionary


def extract_story(story_file):
    if not os.path.isfile(story_file) or not story_file.endswith('.story'):
        return None

    lines = read_text_file(story_file)
    lines = [fix_missing_period(line) for line in lines]

    # separate out article text and highlights

    article_lines = []
    highlights = []

    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    if len(article_lines) == 0 or len(highlights) == 0:
        return

    return Story(' '.join(article_lines), highlights, f_name=story_file)


def tokenize_file(text_file):
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    text = ' '.join(tokenize(text.lower()))

    lines = text.splitlines()
    lines = [line.strip() for line in lines]

    with open(text_file, 'w', encoding='utf-8') as w:
        for line in lines:
            w.write(line + '\n')


def tokenize(text):
    doc = nlp(u'' + text)
    tokens = []
    for token in doc:
        tokens.append(token.text)
    return tokens


def read_text_file(text_file):
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = f.read().lower().splitlines()
    return lines


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line:
        return line
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + " ."


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
