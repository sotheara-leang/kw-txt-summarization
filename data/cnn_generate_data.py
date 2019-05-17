# https://github.com/helmertz/querysum-data/blob/master/convert_rcdata.py

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import hashlib
import io
import os
import re
import sys
import tqdm

import nltk
nltk.download('punkt')

SEP_QUERY = ','
SEP_SUMMARY = '#S#'
SEP_SUMMARY_QUERY = '#Q#'
SEP_ENTITY = ','


class ArticleData:
    def __init__(self, article_text=None, query_to_summaries=None, entities=None):
        self.article_text = article_text
        if query_to_summaries is None:
            self.query_to_summaries = {}
        else:
            self.query_to_summaries = query_to_summaries
        self.entities = entities


class Summaries:
    def __init__(self, first_query_sentence=None, reference_summaries=None, synthetic_summary=None):
        self.first_query_sentence = first_query_sentence
        self.reference_summaries = reference_summaries
        self.synthetic_summary = synthetic_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type=int, default=-1)
    parser.add_argument('--stories_dirs', nargs="*")
    parser.add_argument('--questions_dirs', nargs="*")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--extract', type=int, default=0)     # 0: all data, 1: only valid data

    options = parser.parse_args()

    print(">>> Extracting summarization data...")
    sys.stdout.flush()
    datasets = extract_data(options)
    print("Done")

    print(">>> Saving to output directory...")
    sys.stdout.flush()
    write_data(datasets, options)
    print(">>> Done")


def extract_data(options):
    datasets = {}

    # Look through all question files
    for idx, questions_dir in enumerate(options.questions_dirs):

        print('\n>>> Process %s' % questions_dir)

        for item in os.listdir(questions_dir):

            if os.path.isfile(os.path.join(questions_dir, item)):
                continue

            print('\n>>> Extract questions from %s' % item)

            datasets[item] = {}

            root = os.path.join(questions_dir, item)

            question_counter = 0

            for question_file_name in tqdm.tqdm(os.listdir(root)):

                if not os.path.isfile(os.path.join(root, question_file_name)):
                    continue

                if options.max > 0 and question_counter >= options.max:
                    break

                with io.open(os.path.join(root, question_file_name), 'r', encoding='utf-8') as question_file:
                    question_text = question_file.read()

                url, query, entities = extract_from_question(question_text)

                article_data = datasets[item].get(url)

                if article_data is None:
                    # First time article is processed
                    article_data = ArticleData(entities=entities)
                    datasets[item][url] = article_data

                # Check if summaries for the document-query pair has already been found
                summaries = article_data.query_to_summaries.get(join(query))
                if summaries is not None:
                    continue

                extract_from_story(query, article_data, options.stories_dirs[idx], url, options)

                question_counter += 1

    return datasets


def extract_from_question(question_text):
    lines = question_text.splitlines()

    url = lines[0]
    placeholder = lines[6]
    entity_mapping_lines = lines[8:]

    entity_dictionary = get_entity_dictionary(entity_mapping_lines)

    query = entity_dictionary[placeholder]
    tokenized_query = tokenize(query)
    entities = [entity for entity in entity_dictionary.values()]

    return url, tokenized_query, entities


def get_entity_dictionary(entity_mapping_lines):
    entity_dictionary = {}
    for mapping in entity_mapping_lines:
        entity, name = mapping.split(':', 1)
        entity_dictionary[entity] = name
    return entity_dictionary


def generate_synthetic_summary(document, highlight):
    return [word for word in highlight if word in document]


def extract_from_story(query, article_data, stories_path, url, options):
    # Find original story file which is named using the URL hash
    url_hash = hash_hex(url)
    with io.open(os.path.join(stories_path, '{}.story'.format(url_hash)), 'r', encoding='utf-8') as file:
        raw_article_text = file.read()

    highlight_start_index = raw_article_text.find('@highlight')

    article_text = raw_article_text[:highlight_start_index].strip()
    highlight_text = raw_article_text[highlight_start_index:].strip()

    if len(article_text) == 0:
        # There are stories with only highlights, skip these
        return

    # Extract all highlights
    highlights = re.findall('@highlight\n\n(.*)', highlight_text)

    tokenized_highlights = map(tokenize, highlights)
    tokenized_query_highlights = []

    for highlight in tokenized_highlights:
        if contains_sublist(highlight, query):
            tokenized_query_highlights.append(highlight)

    if len(tokenized_query_highlights) == 0:
        # For now, ignore if sequence of tokens not found in any highlight. It happens for example when query is
        # "American" and highlight contains "Asian-American".
        if options.extract == 1:
            return
        else:
            query = []

    first_query_sentence = get_first_query_sentence(query, article_text)

    synthetic_summary = generate_synthetic_summary(first_query_sentence, tokenized_query_highlights[0] if len(tokenized_query_highlights) > 0 else tokenized_query_highlights)

    reference_summaries = [join(tokenized_highlight) for tokenized_highlight in tokenized_query_highlights]

    summaries = Summaries(join(first_query_sentence), reference_summaries, join(synthetic_summary))

    if article_data.article_text is None:
        article_data.article_text = join(tokenize(article_text))

    article_data.query_to_summaries[join(query)] = summaries


def contains_sublist(list_, sublist):
    for i in range(len(list_)):
        if list_[i:(i + len(sublist))] == sublist:
            return True
    return False


def get_first_query_sentence(query, text):
    # Find sentence containing the placeholder
    sentences = []
    for paragraph in text.splitlines():
        sentences.extend(nltk.sent_tokenize(paragraph))

    for sentence in sentences:
        tokenized_sentence = tokenize(sentence)
        if contains_sublist(tokenized_sentence, query):
            first_query_sentence = tokenized_sentence
            break
    else:
        # Query text not found in document, pick first sentence instead
        first_query_sentence = sentences[0]

    # If ending with a period, remove it, to match most of the highlights
    # (some are however single sentences ending with period)
    if first_query_sentence[-1] == '.':
        first_query_sentence = first_query_sentence[:-1]

    return first_query_sentence


apostrophe_words = {
    "''",
    "'s",
    "'re",
    "'ve",
    "'m",
    "'ll",
    "'d",
    "'em",
    "'n'",
    "'n",
    "'cause",
    "'til",
    "'twas",
    "'till"
}


def lower_and_fix_apostrophe_words(word):
    regex = re.compile("^'\D|'\d+[^s]$")  # 'g | not '90s
    word = word.lower()

    if regex.match(word) and word not in apostrophe_words:
        word = "' " + word[1:]
    return word


def tokenize(text):
    # The Stanford tokenizer may be preferable since it was used for pre-trained GloVe embeddings. However, it appears
    # to be unreasonably slow through the NLTK wrapper.

    tokenized = nltk.tokenize.word_tokenize(text)
    tokenized = [lower_and_fix_apostrophe_words(word) for word in tokenized]
    return tokenized


def join(text):
    return " ".join(text)


def hash_hex(string):
    hash_ = hashlib.sha1()
    hash_.update(string.encode('utf-8'))
    return hash_.hexdigest()


def write_data(datasets, options):
    output_dir = options.output_dir

    for dataset_name, articles in datasets.items():

        # Ignore articles where no query was found
        filtered_articles = [item for item in articles.items() if len(item[1].query_to_summaries) > 0]

        # Get articles ordered by hash values to break possible patterns in ordering
        sorted_articles = sorted(filtered_articles, key=lambda article_tuple: hash_hex(article_tuple[0]))

        output_set_dir = os.path.join(output_dir, dataset_name)

        if not os.path.exists(output_set_dir):
            os.makedirs(output_set_dir)

        article_filename = dataset_name + '.article.txt'
        summary_filename = dataset_name + '.summary.txt'
        keyword_filename = dataset_name + '.keyword.txt'
        entity_filename  = dataset_name + '.entity.txt'

        with io.open(os.path.join(output_set_dir, article_filename), 'w', encoding='utf-8') as article_file, \
                io.open(os.path.join(output_set_dir, summary_filename), 'w', encoding='utf-8') as summary_file, \
                io.open(os.path.join(output_set_dir, keyword_filename), 'w', encoding='utf-8') as keyword_file, \
                io.open(os.path.join(output_set_dir, entity_filename), 'w', encoding='utf-8') as entity_file:

            for url, article_data in tqdm.tqdm(sorted_articles):
                article_text = article_data.article_text
                query_to_summaries = article_data.query_to_summaries
                entities = article_data.entities

                # Ignore if no queries were found for article
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
                    reference_summaries = summaries.reference_summaries
                    for sum_idx, summary in enumerate(reference_summaries):
                        if sum_idx > 0:
                            summary_file.write(SEP_SUMMARY)

                        summary_file.write(summary)

                keyword_file.write('\n')
                summary_file.write('\n')


if __name__ == '__main__':
    main()
