import argparse
from . import util


def add_all_defaults(parser):
    add_data_args(parser)
    add_optimisation_args(parser)
    add_model_args(parser)


def add_optimisation_args(parser):
    # Optimization
    parser.add_argument('--eval-batches', type=int, default=200)
    parser.add_argument('--wait-epochs', type=int, default=5)


def add_data_args(parser):
    # Data defaults
    parser.add_argument('--max-train-types', type=int, default=-1)
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--eval-batch-size', type=int, default=512)


def add_model_args(parser):
    # Model defaults
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=.33)
    parser.add_argument('--checkpoints-path', type=str)


def get_argparser():
    parser = argparse.ArgumentParser(description='LanguageModel')
    parser.add_argument('--seed', type=int, default=7,
                        help='Seed for random algorithms repeatability (default: 7)')
    return parser


def parse_args(parser):
    args = parser.parse_args()
    util.config(args.seed)

    if 'max_train_types' in args and args.max_train_types == -1:
        args.max_train_types = None

    return args
