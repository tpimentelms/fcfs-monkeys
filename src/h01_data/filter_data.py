import os
import sys
import logging
import string
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util.argparser import get_argparser, parse_args, add_data_args
from util.language_characters import get_character_set


def get_args():
    argparser = get_argparser()
    argparser.add_argument("--wikipedia-tokenized-file", type=str,)
    argparser.add_argument("--language", type=str,)

    add_data_args(argparser)
    return parse_args(argparser)


def count_sentences(fname):
    count = 0
    with open(fname, 'r', encoding='utf8') as f:
        for _ in f:
            count += 1
    return count


def is_allowed(word, char_set):
    # return word.isalpha()
    return all(char in char_set for char in word.lower())


def is_integer(x):
    return not (x.isdigit() or (x[0] == '-' and x[1:].isdigit()))


def get_valid_sentence(line, language):
    # exclude words that contain non-letters
    character_set = get_character_set(language)
    # remove punctuation
    line = line.translate(str.maketrans('', '', string.punctuation))
    sentence = [word.lower() for word in list(filter(None, line.strip().split(' ')))]
    # remove numbers
    sentence = list(filter(is_integer, sentence))

    if len(sentence) < 2:
        return None

    # only accept words without extra symbols
    if not all(is_allowed(word, character_set) for word in sentence):
        return None

    return sentence


def write_sentence(tgt_fname, sentence):
    with open(tgt_fname, 'a', encoding='utf8') as f:
        f.write(' '.join(sentence) + '\n')


def filter_data(src_fname, tgt_fname, language):
    n_sentences = count_sentences(src_fname)
    n_skipped = 0

    with open(src_fname, 'r', encoding='utf8') as f:
        for line in tqdm(f, desc='Filtering wiki data',
                         total=n_sentences):
            sentence = get_valid_sentence(line, language)

            if sentence is not None:
                write_sentence(tgt_fname, sentence)
            else:
                n_skipped += 1

    return n_skipped, n_sentences


def main():
    args = get_args()
    logging.info(args)

    n_skipped, n_sentences = filter_data(
        args.wikipedia_tokenized_file, args.data_file,
        args.language)

    print('# skipped:', n_skipped)
    print('# sentences:', n_sentences)


if __name__ == '__main__':
    main()
