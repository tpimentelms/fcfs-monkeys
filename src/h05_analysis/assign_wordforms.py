import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h01_data.process_tokens import load_token_data
from util import util
from util.argparser import get_argparser, parse_args


def get_args():
    argparser = get_argparser()
    argparser.add_argument('--fcfs-samples-file', type=str, required=True)
    argparser.add_argument('--caplan-samples-file', type=str, required=True)
    argparser.add_argument('--caplan-low-temperature-samples-file', type=str, required=True)
    argparser.add_argument('--tokens-file', type=str, required=True)
    argparser.add_argument('--results-file', type=str, required=True)

    args = parse_args(argparser)
    util.config(args.seed)
    return args


def main():
    args = get_args()

    # Load tokens data
    df = load_token_data(args.tokens_file)
    df.sort_values('idx', inplace=True)
    df['frequencies'] = df['count']
    df['natural'] = df['wordform'].apply(str)
    del df['count']
    del df['wordform']

    # Our FCFS monkey model
    fcfs_samples = util.read_data(args.fcfs_samples_file)
    df['fcfs'] = df.idx.apply(lambda x: fcfs_samples[x])

    # Caplan et al's monkey model
    caplan_samples = util.read_data(args.caplan_samples_file)
    df['caplan'] = df.idx.apply(lambda x: caplan_samples[x])

    # Caplan et al's monkey model with low temperatures
    caplan_low_temp_samples = util.read_data(args.caplan_low_temperature_samples_file)
    df['caplan_low_temperature'] = df.idx.apply(lambda x: caplan_low_temp_samples[x])

    df['natural_length'] = df.natural.apply(len)
    df['fcfs_length'] = df.fcfs.apply(len)
    df['caplan_length'] = df.caplan.apply(len)
    df['caplan_low_temperature_length'] = df.caplan_low_temperature.apply(len)

    df.to_csv(args.results_file, sep='\t')


if __name__ == '__main__':
    main()
