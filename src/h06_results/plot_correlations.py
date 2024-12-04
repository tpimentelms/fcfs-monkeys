import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import constants
from util import util
from util.argparser import get_argparser, parse_args

aspect = {
    'height': 7,
    'font_scale': 1.5,
    'labels': True,
    'name_suffix': '',
    'ratio': 1.625,
}
sns.set_palette("Set2")
sns.set_context("notebook", font_scale=aspect['font_scale'])
mpl.rc('font', family='serif', serif='Times New Roman')


def get_args():
    argparser = get_argparser()
    argparser.add_argument('--checkpoints-path', type=str, required=True)
    argparser.add_argument('--results-path', type=str, required=True)
    argparser.add_argument('--use-low-temperature', default=False, action='store_true')

    args = parse_args(argparser)
    util.config(args.seed)
    return args


def read_seed_results(checkpoints_path, seed):
    results_files = [
        os.path.join(checkpoints_path, language, 'seed_%02d' % seed, 'compiled_results.tsv')
        for language in constants.LANGUAGES
    ]
    dfs = [
        pd.read_csv(file, sep='\t')
        for file in results_files
    ]

    return pd.concat(dfs)


def read_results(checkpoints_path):
    dfs = []
    for seed in range(10):
        df = read_seed_results(checkpoints_path, seed)
        df['seed'] = seed
        dfs += [df]

    return pd.concat(dfs)


def plot_frequency_length(df, results_path, use_low_temperature=False):
    df = df.copy()

    df['Language'] = df['language'].apply(lambda x: constants.LANG_NAMES[x])
    df['Natural'] = df['natural_frequency_corr']
    df['FCFS'] = df['fcfs_frequency_corr']
    df['PolyFCFS'] = df['polyfcfs_frequency_corr']

    if use_low_temperature:
        df['IID'] = df['caplan_low_temp_frequency_corr']
        df['PolyIID'] = df['polycaplan_low_temp_frequency_corr']
        fname = os.path.join(results_path, 'plot_frequency_length--low_temperature.pdf')
    else:
        df['IID'] = df['caplan_frequency_corr']
        df['PolyIID'] = df['polycaplan_frequency_corr']
        fname = os.path.join(results_path, 'plot_frequency_length.pdf')

    df = pd.melt(df, id_vars=['Language', 'seed'], var_name='Text', value_name='Correlation',
                 value_vars=['Natural', 'FCFS', 'PolyFCFS', 'IID', 'PolyIID'])
    df['Correlation (%)'] = df['Correlation'] * 100

    fig = plt.figure()
    ax = sns.barplot(x="Language", y="Correlation (%)", hue="Text", data=df)
    ax.plot([2], [-5], '-', color='none', label=' ')

    # plt.ylim([-31, 1])
    plt.xlim([-.5, 5.5])
    plt.ylim([-35, 1.3])

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: constants.LEGEND_ORDER[t[0]]))
    # plt.legend(handles, labels, loc='lower center', ncol=3, handletextpad=.5, columnspacing=1.2)
    plt.legend(handles, labels, loc='lower center', ncol=3, handletextpad=.3, columnspacing=.75)

    plt.xticks(rotation=15)

    fig.savefig(fname, bbox_inches='tight')


def plot_polysemy_length(df, results_path, use_low_temperature=False):
    df = df.copy()

    df['Language'] = df['language'].apply(lambda x: constants.LANG_NAMES[x])
    df['Natural'] = df['natural_polysemy_corr']
    df['FCFS'] = df['fcfs_polysemy_corr']
    df['PolyFCFS'] = df['polyfcfs_polysemy_corr']

    if use_low_temperature:
        df['IID'] = df['caplan_low_temp_polysemy_corr']
        df['PolyIID'] = df['polycaplan_low_temp_polysemy_corr']
        fname = os.path.join(results_path, 'plot_polysemy_length--low_temperature.pdf')
    else:
        df['IID'] = df['caplan_polysemy_corr']
        df['PolyIID'] = df['polycaplan_polysemy_corr']
        fname = os.path.join(results_path, 'plot_polysemy_length.pdf')

    df = pd.melt(df, id_vars=['Language', 'seed'], var_name='Text', value_name='Correlation',
                 value_vars=['Natural', 'FCFS', 'PolyFCFS', 'IID', 'PolyIID'])
    df['Correlation (%)'] = df['Correlation'] * 100

    fig = plt.figure()
    ax = sns.barplot(x="Language", y="Correlation (%)", hue="Text", data=df)
    ax.plot([2], [-5], '-', color='none', label=' ')
    # mpl.lines.Line2D([0], [0], color="none", label=' haha')

    # plt.ylim([-26, 1.5])
    plt.xlim([-.5, 5.5])
    # plt.ylim([-29, 1.3])
    plt.ylim([-42, 1.3])
    # plt.ylim([-27, 2.5])

    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: constants.LEGEND_ORDER[t[0]]))
    plt.legend(handles, labels, loc='lower center', ncol=3, handletextpad=.3, columnspacing=.75)
    plt.xticks(rotation=15)

    fig.savefig(fname, bbox_inches='tight')


def plot_freq_length(checkpoints_path, results_path, use_low_temperature=False):
    df = read_results(checkpoints_path)

    plot_frequency_length(df, results_path, use_low_temperature)
    plot_polysemy_length(df, results_path, use_low_temperature)


def main():
    args = get_args()

    plot_freq_length(args.checkpoints_path, args.results_path, args.use_low_temperature)


if __name__ == '__main__':
    main()
