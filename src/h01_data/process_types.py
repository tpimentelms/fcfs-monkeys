import os
import sys
import logging
import numpy as np
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h01_data.alphabet import Alphabet
from h01_data.filter_data import count_sentences
from util.argparser import get_argparser, parse_args, add_data_args
from util import util


def get_args():
    argparser = get_argparser()
    argparser.add_argument("--wikipedia-tokenized-file", type=str,)
    argparser.add_argument("--max-types", type=int, required=True)
    argparser.add_argument("--n-folds", type=int, default=10,)
    add_data_args(argparser)
    return parse_args(argparser)


def get_fold_splits(n_sentences, n_folds, shuffle_splits=True):
    splits = np.arange(n_sentences)
    if shuffle_splits:
        np.random.shuffle(splits)
    splits = np.array_split(splits, n_folds)
    splits = {x: i for i, fold in enumerate(splits) for x in fold}
    return splits


def process_line(line, word_freq, alphabet):
    sentence = line.strip().replace('-', ' ').split(' ')

    for word in sentence:
        word = word.lower()
        alphabet.add_word(word)

        word_freq[word] = word_freq.get(word, 0) + 1


def get_types(src_fname, alphabet, n_sentences):
    word_freq = {}
    with open(src_fname, 'r', encoding='utf8') as f:
        for line in tqdm(f, desc='Processing wiki data',
                         total=n_sentences):
            process_line(line, word_freq, alphabet)
    return word_freq


def filter_types(word_freq, max_types):
    word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return word_freq[:max_types]


def split_types(word_freq, n_folds):
    splits = get_fold_splits(len(word_freq), n_folds)
    word_folds = [[] for _ in range(n_folds)]
    for idx, fold in splits.items():
        word_folds[fold] += [word_freq[idx]]

    return [dict(fold) for fold in word_folds]


def process(src_fname, tgt_fname, n_folds, max_types):
    alphabet = Alphabet()
    n_sentences = count_sentences(src_fname)

    word_freq = get_types(src_fname, alphabet, n_sentences)
    n_types_raw = len(word_freq)
    word_freq = filter_types(word_freq, max_types)
    n_types = len(word_freq)

    word_folds = split_types(word_freq, n_folds)
    util.write_data(tgt_fname, (word_folds, alphabet))

    print('# unique chars:', len(alphabet))
    print('# types raw: %d' % n_types_raw)
    print('# types filtered: %d' % n_types)


def main():
    args = get_args()
    logging.info(args)

    process(args.wikipedia_tokenized_file, args.data_file,
            args.n_folds, args.max_types)


if __name__ == '__main__':
    main()
