import os
import sys
from tqdm import tqdm
from ordered_set import OrderedSet

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h01_data.process_tokens import load_token_data
from h02_polysemy.get_polyassign_code import load_polysemy_codes
from h03_learn.dataset import get_data_loaders
from h03_learn.train import load_model
from util import util
from util.argparser import get_argparser, parse_args


def get_args():
    argparser = get_argparser()
    argparser.add_argument('--checkpoint-path', type=str, required=True)
    argparser.add_argument('--tokens-file', type=str, required=True)
    argparser.add_argument('--types-file', type=str, required=True)
    argparser.add_argument('--polyassign-code-file', type=str, required=True)
    argparser.add_argument('--samples-file', type=str, required=True)
    argparser.add_argument('--with-repetition', action='store_true', default=False)
    argparser.add_argument('--temperature', type=float, required=True)
    return parse_args(argparser)


# Wrapper of both an OrderedSet and a List
class SampleList:
    def __init__(self, with_repetition=False):
        self.with_repetition = with_repetition

        if self.with_repetition:
            self.samples = []
        else:
            self.samples = OrderedSet()

    def append(self, new_samples):
        if self.with_repetition:
            self.samples += new_samples
        else:
            self.samples.update(new_samples)

    def limit_size(self, max_samples):
        if len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

    def tolist(self):
        if self.with_repetition:
            samples = self.samples
        else:
            samples = list(self.samples)

        return samples

    def __iter__(self):
        for x in self.samples:
            yield x

    def __len__(self):
        return len(self.samples)


def get_samples(model, n_samples, alphabet, temperature, with_repetition=False):
    sampling_rate = 1000
    samples = SampleList(with_repetition=with_repetition)

    print('Sampling %d instances' % n_samples)
    with tqdm(total=n_samples) as pbar:
        while len(samples) < n_samples:
            new_samples = model.sample(alphabet, sampling_rate, temperature=temperature)
            samples.append(new_samples)

            avg_length = sum(len(x) for x in samples) / len(samples)

            pbar.n = len(samples)
            pbar.set_description('Avg length %.4f' % (avg_length))

    samples.limit_size(n_samples)
    return samples.tolist()


def main():
    args = get_args()
    folds = [[], [], list(range(10))]

    model = load_model(args.checkpoint_path)

    _, _, _, alphabet = get_data_loaders(args.types_file, folds)
    tokens = load_token_data(args.tokens_file)
    poly_codes = load_polysemy_codes(args.polyassign_code_file)

    n_tokens = tokens.shape[0]
    n_poly = len(set(poly_codes))
    print('Unique Natural Types: %d\tUnique Polysemy types: %d' % (n_tokens, n_poly))
    n_samples = max(n_tokens, n_poly)

    samples = get_samples(model, n_samples, alphabet, temperature=args.temperature,
                          with_repetition=args.with_repetition)

    util.write_data(args.samples_file, samples)


if __name__ == '__main__':
    main()
