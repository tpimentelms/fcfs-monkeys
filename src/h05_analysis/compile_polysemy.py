import os
import sys
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import util
from util.argparser import get_argparser, parse_args


def get_args():
    argparser = get_argparser()
    argparser.add_argument('--fcfs-samples-file', type=str, required=True)
    argparser.add_argument('--caplan-samples-file', type=str, required=True)
    argparser.add_argument('--caplan-low-temperature-samples-file', type=str, required=True)
    argparser.add_argument('--results-ent-polyassign-file', type=str, required=True)
    argparser.add_argument('--results-ent-natural-file', type=str, required=True)
    argparser.add_argument('--results-compiled-polyassign-file', type=str, required=True)
    argparser.add_argument('--results-compiled-natural-file', type=str, required=True)

    args = parse_args(argparser)
    util.config(args.seed)
    return args


def expand_dataframe(df, fcfs_samples, caplan_samples, caplan_low_temp_samples):
    df['frequencies'] = df['count']
    df['natural'] = df['wordform'].apply(str)
    del df['count']
    del df['wordform']

    df['fcfs'] = df.idx.apply(lambda x: fcfs_samples[x])
    df['caplan'] = df.idx.apply(lambda x: caplan_samples[x])
    df['caplan_low_temperature'] = df.idx.apply(lambda x: caplan_low_temp_samples[x])

    df['natural_length'] = df.natural.apply(len)
    df['fcfs_length'] = df.fcfs.apply(len)
    df['caplan_length'] = df.caplan.apply(len)
    df['caplan_low_temperature_length'] = df.caplan_low_temperature.apply(len)

    return df


def main():
    # pylint: disable=too-many-locals
    args = get_args()

    # Our FCFS monkey model
    fcfs_samples = util.read_data(args.fcfs_samples_file)
    # Caplan et al's monkey model
    caplan_samples = util.read_data(args.caplan_samples_file)
    caplan_low_temp_samples = util.read_data(args.caplan_low_temperature_samples_file)

    # Extend polysemy results
    df_natural = pd.read_csv(args.results_ent_natural_file, sep='\t')
    df_natural = expand_dataframe(df_natural, fcfs_samples, caplan_samples, caplan_low_temp_samples)
    df_natural.to_csv(args.results_compiled_natural_file, sep='\t', index=False)

    # Extend polysemy results
    df_polyassign = pd.read_csv(args.results_ent_polyassign_file, sep='\t')
    df_polyassign['wordform'] = df_polyassign['wordform'].apply(str)
    df_polyassign = expand_dataframe(df_polyassign, fcfs_samples, caplan_samples,
                                     caplan_low_temp_samples)
    df_polyassign.to_csv(args.results_compiled_polyassign_file, sep='\t', index=False)


if __name__ == '__main__':
    main()
