import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from nltk.corpus import wordnet as wn

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import util
from util.argparser import get_argparser, parse_args


def get_args():
    argparser = get_argparser()
    argparser.add_argument('--language', type=str, required=True)
    argparser.add_argument('--results-freq-codes-file', type=str, required=True)
    argparser.add_argument('--results-ent-polyassign-file', type=str, required=True)
    argparser.add_argument('--results-ent-natural-file', type=str, required=True)
    argparser.add_argument('--results-compiled-file', type=str, required=True)

    args = parse_args(argparser)
    util.config(args.seed)
    return args


def str_for_table(df, min_count, var1='poly_cov', var2='len'):

    corr, pvalue = stats.spearmanr(
        df.loc[df['count'] >= min_count, var1],
        df.loc[df['count'] >= min_count, var2])

    if pvalue < 0.01:
        pvalue_str = '$\\ddagger$'
    elif pvalue < 0.05:
        pvalue_str = '$\\dagger$'
    else:
        pvalue_str = ''
    return '%.2f%s' % (corr, pvalue_str)


def get_frequency_length_correlations(df, df_poly):
    # natural correlations
    natural_frequency_correlations = stats.spearmanr(df['frequencies'],
                                                     df['natural_length'])
    # fcfs correlations
    fcfs_frequency_correlations = stats.spearmanr(df['frequencies'],
                                                  df['fcfs_length'])
    polyfcfs_frequency_correlations = stats.spearmanr(df_poly['frequencies'],
                                                      df_poly['fcfs_length'])

    # iid correlations
    df_iid = df.groupby('caplan').agg({'caplan_length': 'mean',
                                       'frequencies': 'sum'})
    caplan_frequency_correlations = stats.spearmanr(df_iid['frequencies'],
                                                    df_iid['caplan_length'], nan_policy='omit')
    df_poly_iid = df_poly.groupby('caplan').agg({'caplan_length': 'mean',
                                                 'frequencies': 'sum'})
    polycaplan_frequency_correlations = stats.spearmanr(
        df_poly_iid['frequencies'], df_poly_iid['caplan_length'], nan_policy='omit')

    # iid with low temperature correlations
    df_iid = df.groupby('caplan_low_temperature') \
        .agg({'caplan_low_temperature_length': 'mean', 'frequencies': 'sum'})
    caplan_low_temp_frequency_correlations = stats.spearmanr(
        df_iid['frequencies'], df_iid['caplan_low_temperature_length'], nan_policy='omit')
    df_poly_iid = df_poly.groupby('caplan_low_temperature') \
        .agg({'caplan_low_temperature_length': 'mean', 'frequencies': 'sum'})
    polycaplan_low_temp_frequency_correlations = stats.spearmanr(
        df_poly_iid['frequencies'], df_poly_iid['caplan_low_temperature_length'], nan_policy='omit')

    return natural_frequency_correlations, fcfs_frequency_correlations, \
        polyfcfs_frequency_correlations, caplan_frequency_correlations, \
        polycaplan_frequency_correlations, caplan_low_temp_frequency_correlations, \
        polycaplan_low_temp_frequency_correlations


def get_wordnet_synsets(df, lang_code):
    df['natural'] = df['natural'].apply(str)
    df['wn_synset'] = df['natural'].apply(
        lambda x: len(wn.synsets(x, lang=lang_code)) if lang_code else -1)
    return df


def filter_min_count(df_nat, df_polyassign, min_count):
    df_nat = df_nat[df_nat['frequencies'] >= min_count]
    df_polyassign = df_polyassign[df_polyassign['frequencies'] >= min_count]

    return df_nat, df_polyassign


def merge_homophone_entropy(df, wordform_col='caplan', length_col='caplan_length',
                            polysemy_col='poly_var'):
    df = df.copy()
    df['full_frequency'] = df.groupby(wordform_col)['frequencies'].transform('sum')
    df['homophone_probability'] = df['frequencies'] / df['full_frequency']

    df['homophone_weighted_polysemy'] = df['homophone_probability'] * df[polysemy_col]
    df['homophone_surprisal'] = - df['homophone_probability'].apply(np.log)
    df['homophone_weighted_surprisal'] = df['homophone_probability'] * df['homophone_surprisal']

    df_iid = df.groupby(wordform_col).agg({length_col: 'mean',
                                           'frequencies': 'sum',
                                           'homophone_weighted_polysemy': 'sum',
                                           'homophone_weighted_surprisal': 'sum'})
    df_iid[polysemy_col] = \
        df_iid['homophone_weighted_surprisal'] + df_iid['homophone_weighted_polysemy']

    return df_iid.reset_index()


def get_polysemy_length_correlations(df_nat, df_polyassign,
                                     wn_eval, polysemy_col='poly_var'):
    # WordNet correlations
    wornet_vs_polysemy_corr = stats.spearmanr(wn_eval['wn_synset'], wn_eval[polysemy_col])
    natural_wordnet_len_corr = stats.spearmanr(wn_eval['wn_synset'], wn_eval['natural_length'])

    # Polysemy correlations
    natural_polysemy_corr = stats.spearmanr(df_nat[polysemy_col],
                                            df_nat['natural_length'])
    fcfs_polysemy_corr = stats.spearmanr(df_nat[polysemy_col],
                                         df_nat['fcfs_length'])
    polyfcfs_polysemy_corr = stats.spearmanr(df_polyassign[polysemy_col],
                                             df_polyassign['fcfs_length'])

    # Iid polysemy correlations
    df_nat_iid = merge_homophone_entropy(
        df_nat, 'caplan', 'caplan_length', polysemy_col)
    caplan_polysemy_corr = stats.spearmanr(
        df_nat_iid[polysemy_col], df_nat_iid['caplan_length'], nan_policy='omit')
    df_polyassign_iid = merge_homophone_entropy(
        df_polyassign, 'caplan', 'caplan_length', polysemy_col)
    polycaplan_polysemy_corr = stats.spearmanr(
        df_polyassign_iid[polysemy_col], df_polyassign_iid['caplan_length'], nan_policy='omit')

    # Iid low temperature polysemy correlations
    df_nat_iid = merge_homophone_entropy(
        df_nat, 'caplan_low_temperature', 'caplan_low_temperature_length', polysemy_col)
    caplan_low_temp_polysemy_corr = stats.spearmanr(
        df_nat_iid[polysemy_col], df_nat_iid['caplan_low_temperature_length'], nan_policy='omit')
    df_polyassign_iid = merge_homophone_entropy(
        df_polyassign, 'caplan_low_temperature', 'caplan_low_temperature_length', polysemy_col)
    polycaplan_low_temp_polysemy_corr = stats.spearmanr(
        df_polyassign_iid[polysemy_col], df_polyassign_iid['caplan_low_temperature_length'],
        nan_policy='omit')

    return wornet_vs_polysemy_corr, natural_wordnet_len_corr, \
        natural_polysemy_corr, fcfs_polysemy_corr, polyfcfs_polysemy_corr, \
        caplan_polysemy_corr, polycaplan_polysemy_corr, \
        caplan_low_temp_polysemy_corr, polycaplan_low_temp_polysemy_corr


def main():
    # pylint: disable=too-many-locals
    args = get_args()

    lang_codes = {
        'en': 'eng',
        'fi': 'fin',
        'pt': 'por',
        'id': 'ind',
        'he': None,
        'tr': None,
        'simple': 'eng',
    }

    # Get natural and fcfs code lengths
    df_length = pd.read_csv(args.results_freq_codes_file, sep='\t')
    del df_length['Unnamed: 0']

    # Get polysemy results
    df_nat = pd.read_csv(args.results_ent_natural_file, sep='\t')
    df_polyassign = pd.read_csv(args.results_ent_polyassign_file, sep='\t')

    # Get frequency--length correlations
    natural_frequency_corr, fcfs_frequency_corr, polyfcfs_frequency_corr, \
        caplan_frequency_corr, polycaplan_frequency_corr, \
        caplan_low_temp_frequency_corr, polycaplan_low_temp_frequency_corr = \
        get_frequency_length_correlations(df_length, df_polyassign)

    # Filter words with less than 10 ocurrances
    df_nat, df_polyassign = \
        filter_min_count(df_nat, df_polyassign, min_count=10)

    # Get number of sense in Wordnet
    df_nat = get_wordnet_synsets(df_nat, lang_codes[args.language])
    wn_eval = df_nat[df_nat['wn_synset'] > 1]

    # Get polysemy--length correlations
    wornet_vs_polysemy_corr, natural_wordnet_len_corr, natural_polysemy_corr, \
        fcfs_polysemy_corr, polyfcfs_polysemy_corr, \
        caplan_polysemy_corr, polycaplan_polysemy_corr, \
        caplan_low_temp_polysemy_corr, polycaplan_low_temp_polysemy_corr = \
        get_polysemy_length_correlations(df_nat, df_polyassign, wn_eval)

    results = {
        'language': [args.language],
        'n_types_frequency_experiment': df_length.shape[0],
        'n_types_wordnet': wn_eval.shape[0],
        'n_types_polysemy_natural': df_nat.shape[0],
        'n_types_polysemy_polyassign': df_polyassign.shape[0],
        'natural_frequency_corr': [natural_frequency_corr.correlation],
        'natural_frequency_corr--pvalue': [natural_frequency_corr.pvalue],
        'fcfs_frequency_corr': [fcfs_frequency_corr.correlation],
        'fcfs_frequency_corr--pvalue': [fcfs_frequency_corr.pvalue],
        'polyfcfs_frequency_corr': [polyfcfs_frequency_corr.correlation],
        'polyfcfs_frequency_corr--pvalue': [polyfcfs_frequency_corr.pvalue],
        'caplan_frequency_corr': [caplan_frequency_corr.correlation],
        'caplan_frequency_corr--pvalue': [caplan_frequency_corr.pvalue],
        'polycaplan_frequency_corr': [polycaplan_frequency_corr.correlation],
        'polycaplan_frequency_corr--pvalue': [polycaplan_frequency_corr.pvalue],
        'caplan_low_temp_frequency_corr': [caplan_low_temp_frequency_corr.correlation],
        'caplan_low_temp_frequency_corr--pvalue': [caplan_low_temp_frequency_corr.pvalue],
        'polycaplan_low_temp_frequency_corr': [polycaplan_low_temp_frequency_corr.correlation],
        'polycaplan_low_temp_frequency_corr--pvalue': [polycaplan_low_temp_frequency_corr.pvalue],
        'wornet_vs_polysemy_corr': [wornet_vs_polysemy_corr.correlation],
        'wornet_vs_polysemy_corr--pvalue': [wornet_vs_polysemy_corr.pvalue],
        'natural_wordnet_len_corr': [natural_wordnet_len_corr.correlation],
        'natural_wordnet_len_corr--pvalue': [natural_wordnet_len_corr.pvalue],
        'natural_polysemy_corr': [natural_polysemy_corr.correlation],
        'natural_polysemy_corr--pvalue': [natural_polysemy_corr.pvalue],
        'fcfs_polysemy_corr': [fcfs_polysemy_corr.correlation],
        'fcfs_polysemy_corr--pvalue': [fcfs_polysemy_corr.pvalue],
        'polyfcfs_polysemy_corr': [polyfcfs_polysemy_corr.correlation],
        'polyfcfs_polysemy_corr--pvalue': [polyfcfs_polysemy_corr.pvalue],
        'caplan_polysemy_corr': [caplan_polysemy_corr.correlation],
        'caplan_polysemy_corr--pvalue': [caplan_polysemy_corr.pvalue],
        'polycaplan_polysemy_corr': [polycaplan_polysemy_corr.correlation],
        'polycaplan_polysemy_corr--pvalue': [polycaplan_polysemy_corr.pvalue],
        'caplan_low_temp_polysemy_corr': [caplan_low_temp_polysemy_corr.correlation],
        'caplan_low_temp_polysemy_corr--pvalue': [caplan_low_temp_polysemy_corr.pvalue],
        'polycaplan_low_temp_polysemy_corr': [polycaplan_low_temp_polysemy_corr.correlation],
        'polycaplan_low_temp_polysemy_corr--pvalue': [polycaplan_low_temp_polysemy_corr.pvalue],
    }
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.results_compiled_file, sep='\t', index=False)


if __name__ == '__main__':
    main()
