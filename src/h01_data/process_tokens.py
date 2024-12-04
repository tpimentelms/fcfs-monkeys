import os
import sys
import logging
import pandas as pd
from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h01_data.filter_data import count_sentences
from util.argparser import get_argparser, parse_args, add_data_args


def get_args():
    argparser = get_argparser()
    argparser.add_argument("--wikipedia-tokenized-file", type=str,)
    add_data_args(argparser)
    return parse_args(argparser)


def load_token_data(fname):
    return pd.read_csv(fname, sep='\t')


def process_line(line):
    return line.strip().split(' ')


def process_data(src_fname, n_sentences):
    tokens = []
    with open(src_fname, 'r', encoding='utf8') as f:
        for line in tqdm(f, desc='Processing wiki data', total=n_sentences):
            tokens += process_line(line)
    return tokens


def count_tokens(tokens):
    token_info = {}
    for token in tokens:
        if token not in token_info:
            token_info[token] = {
                'idx': len(token_info),
                'wordform': token,
                'natural_length': len(token),
                'count': 0,
            }
        token_info[token]['count'] += 1

    return token_info


def process(src_fname, tgt_fname):
    n_sentences = count_sentences(src_fname)

    tokens = process_data(src_fname, n_sentences)
    token_info = count_tokens(tokens)
    print('# tokens:', len(tokens))
    print('# types:', len(token_info))

    df = pd.DataFrame(token_info.values())
    df.to_csv(tgt_fname, sep='\t', index=False)


def main():
    args = get_args()
    logging.info(args)

    process(args.wikipedia_tokenized_file, args.data_file)


if __name__ == '__main__':
    main()
