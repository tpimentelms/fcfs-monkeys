import os
import sys
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import constants
from util import util
from util.argparser import get_argparser, parse_args


def get_args():
    argparser = get_argparser()
    argparser.add_argument('--checkpoints-path', type=str, required=True)
    argparser.add_argument('--use-low-temperature', default=False, action='store_true')

    args = parse_args(argparser)
    util.config(args.seed)
    return args


def str_for_table(corr, pvalue):
    if pvalue < 0.01:
        pvalue_str = '$\\ddagger$'
    elif pvalue < 0.05:
        pvalue_str = '$\\dagger$'
    else:
        pvalue_str = ''
    return '%.2f%s' % (corr, pvalue_str)


def print_table1(df):
    table_str = '%s & %s & %s & %s & %s & %s \\\\'

    df_mean = df.groupby('language').agg('mean').reset_index()

    natural_pvalue = util.permutation_test(df.natural_frequency_corr.to_numpy())
    fcfs_pvalue = util.permutation_test(df.fcfs_frequency_corr.to_numpy())
    polyfcfs_pvalue = util.permutation_test(df.polyfcfs_frequency_corr.to_numpy())
    caplan_pvalue = util.permutation_test(df.caplan_frequency_corr.to_numpy())
    polycaplan_pvalue = util.permutation_test(df.polycaplan_frequency_corr.to_numpy())

    language = constants.LANG_NAMES[df_mean['language'].item()]
    natural_corr = str_for_table(
        df_mean['natural_frequency_corr'].item(), natural_pvalue)
    fcfs_corr = str_for_table(
        df_mean['fcfs_frequency_corr'].item(), fcfs_pvalue)
    polyfcfs_corr = str_for_table(
        df_mean['polyfcfs_frequency_corr'].item(), polyfcfs_pvalue)
    caplan_corr = str_for_table(
        df_mean['caplan_frequency_corr'].item(), caplan_pvalue)
    polycaplan_corr = str_for_table(
        df_mean['polycaplan_frequency_corr'].item(), polycaplan_pvalue)

    print(table_str % (language, natural_corr, fcfs_corr, polyfcfs_corr,
                       caplan_corr, polycaplan_corr))


def print_table2(df):
    table_str = '%s & %d & %s & %s & %s \\\\'

    df_mean = df.groupby('language').agg('mean').reset_index()

    language = constants.LANG_NAMES[df_mean['language'].item()]
    if language in ['Hebrew', 'Turkish']:
        n_types_wn = df_mean['n_types_polysemy_natural'].item()
    else:
        n_types_wn = df_mean['n_types_wordnet'].item()

    wordnet_vs_polysemy_pvalue = util.permutation_test(df.wornet_vs_polysemy_corr.to_numpy()) \
        if not df.wornet_vs_polysemy_corr.isna().any() else float('nan')
    natural_wordnet_len_pvalue = util.permutation_test(df.natural_wordnet_len_corr.to_numpy()) \
        if not df.natural_wordnet_len_corr.isna().any() else float('nan')
    natural_polysemy_pvalue = util.permutation_test(df.natural_polysemy_corr.to_numpy()) \
        if not df.natural_polysemy_corr.isna().any() else float('nan')

    wornet_vs_polysemy_corr = str_for_table(
        df_mean['wornet_vs_polysemy_corr'].item(), wordnet_vs_polysemy_pvalue)
    natural_wordnet_len_corr = str_for_table(
        df_mean['natural_wordnet_len_corr'].item(), natural_wordnet_len_pvalue)
    natural_polysemy_corr = str_for_table(
        df_mean['natural_polysemy_corr'].item(), natural_polysemy_pvalue)

    print(table_str %
          (language, n_types_wn, natural_wordnet_len_corr,
           natural_polysemy_corr, wornet_vs_polysemy_corr))


def print_table3(df):
    table_str = '%s & %s & %s & %s & %s & %s \\\\'

    df_mean = df.groupby('language').agg('mean').reset_index()

    natural_pvalue = util.permutation_test(df.natural_polysemy_corr.to_numpy())
    fcfs_pvalue = util.permutation_test(df.fcfs_polysemy_corr.to_numpy())
    polyfcfs_pvalue = util.permutation_test(df.polyfcfs_polysemy_corr.to_numpy())
    caplan_pvalue = util.permutation_test(df.caplan_polysemy_corr.to_numpy())
    polycaplan_pvalue = util.permutation_test(df.polycaplan_polysemy_corr.to_numpy())

    language = constants.LANG_NAMES[df_mean['language'].item()]
    natural_polysemy_corr = str_for_table(
        df_mean['natural_polysemy_corr'].item(), natural_pvalue)
    fcfs_polysemy_corr = str_for_table(
        df_mean['fcfs_polysemy_corr'].item(), fcfs_pvalue)
    polyfcfs_polysemy_corr = str_for_table(
        df_mean['polyfcfs_polysemy_corr'].item(), polyfcfs_pvalue)
    caplan_polysemy_corr = str_for_table(
        df_mean['caplan_polysemy_corr'].item(), caplan_pvalue)
    polycaplan_polysemy_corr = str_for_table(
        df_mean['polycaplan_polysemy_corr'].item(), polycaplan_pvalue)

    print(table_str % (language, natural_polysemy_corr,
                       fcfs_polysemy_corr, polyfcfs_polysemy_corr,
                       caplan_polysemy_corr, polycaplan_polysemy_corr))


def print_table4(df):
    table_str = '%s & %d & %d & %d & %d \\\\'

    df_mean = df.groupby('language').agg('mean').reset_index()
    language = constants.LANG_NAMES[df_mean['language'].item()]

    n_types_frequency_experiment = df_mean['n_types_frequency_experiment'].item()
    n_types_wordnet = df_mean['n_types_wordnet'].item()
    n_types_polysemy_natural = df_mean['n_types_polysemy_natural'].item()
    n_types_polysemy_polyassign = df_mean['n_types_polysemy_polyassign'].item()

    print(table_str % (language, n_types_frequency_experiment, n_types_wordnet,
                       n_types_polysemy_natural, n_types_polysemy_polyassign, ))


def get_language_results(language, checkpoints_path, use_low_temperature=False):
    dfs = []
    for seed in range(10):
        results_compiled_file = os.path.join(
            checkpoints_path, language, 'seed_%02d' % seed, 'compiled_results.tsv')
        df = pd.read_csv(results_compiled_file, sep='\t')
        df['seed'] = seed

        if use_low_temperature:
            df['caplan_frequency_corr'] = df['caplan_low_temp_frequency_corr']
            df['polycaplan_frequency_corr'] = df['polycaplan_low_temp_frequency_corr']
            df['caplan_polysemy_corr'] = df['caplan_low_temp_polysemy_corr']
            df['polycaplan_polysemy_corr'] = df['polycaplan_low_temp_polysemy_corr']

        dfs += [df]

    return pd.concat(dfs)


def print_results(language, checkpoints_path, use_low_temperature=False):
    df = get_language_results(language, checkpoints_path, use_low_temperature)

    pvalue = [util.permutation_test(
        (df.natural_frequency_corr - df[column]).to_numpy())
        for column in ['fcfs_frequency_corr',
                       'polyfcfs_frequency_corr',
                       'caplan_frequency_corr',
                       'polycaplan_frequency_corr']
    ]
    print('%s. Natural vs All frequency--length: %.4f' % (language, max(pvalue)))
    pvalue = [util.permutation_test(
        (df.natural_polysemy_corr - df[column]).to_numpy())
        for column in ['fcfs_polysemy_corr',
                       'polyfcfs_polysemy_corr',
                       'caplan_polysemy_corr',
                       'polycaplan_polysemy_corr']
    ]
    print('%s. Natural vs All polysemy--length: %.4f' % (language, max(pvalue)))
    pvalues = [util.permutation_test(
        (df.fcfs_frequency_corr - df[column]).to_numpy())
        for column in ['caplan_frequency_corr',
                       'polycaplan_frequency_corr']
    ]
    print('%s. FCFS vs IID or Poly-IID frequency--length: %.4f' % (language, max(pvalues)))
    pvalues = [util.permutation_test(
        (df.polyfcfs_polysemy_corr - df[column]).to_numpy())
        for column in ['caplan_frequency_corr',
                       'polycaplan_polysemy_corr']
    ]
    print('%s. Poly-FCFS vs IID or Poly-IID polysemy--length: %.4f' % (language, max(pvalues)))

    print_table1(df)
    print_table2(df)
    print_table3(df)
    print_table4(df)
    print()


def main():
    args = get_args()

    for language in constants.LANGUAGES:
        print_results(language, args.checkpoints_path, args.use_low_temperature)

    # print_results('simple', args.checkpoints_path)


if __name__ == '__main__':
    main()
