import os
import sys
from collections import Counter
from tqdm import tqdm
import numpy as np
from scipy import stats
from sklearn.metrics import pairwise
from sklearn import metrics

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import util
from util.argparser import get_argparser, parse_args


def get_args():
    argparser = get_argparser()
    argparser.add_argument('--polyassign-file', type=str, required=True)
    argparser.add_argument('--emb-dir', type=str, required=True)
    argparser.add_argument('--pca-file', type=str, required=True)

    args = parse_args(argparser)
    util.config(args.seed)
    return args


def load_polysemy_codes(polysemy_file):
    ftemp = polysemy_file.replace('.pckl', '--temp.pckl')
    if util.is_file(polysemy_file):
        codes = util.read_data(polysemy_file)
    elif util.is_file(ftemp):
        codes = util.read_data(ftemp)
        print('Warning! Printing temp results')
    else:
        raise ValueError('No polysemy data to load')

    codes = codes[codes != -1]
    return codes


class IncrementalDist:
    _distances = None
    batch_last = -1

    def __init__(self, embs, batch_size):
        self.embs = embs
        self.batch_size = batch_size

    def mindist(self, idx):
        if self.batch_last < idx:
            self.batch_dists(idx)

        dist_new = self._dists_new[
            idx - self.batch_first, :idx - self.batch_first]
        dist_old = self._dist_old[
            idx - self.batch_first]

        # import ipdb; ipdb.set_trace()
        if (dist_old <= dist_new).all():
            argmin = self._argmin_old[idx - self.batch_first]
            dist = dist_old
        else:
            argmin = dist_new.argmin() + self.batch_first
            dist = dist_new.min()

        return argmin, dist

    def batch_dists(self, idx):
        self.batch_first = idx
        self.batch_last = idx + self.batch_size - 1

        self._argmin_old, self._dist_old = metrics.pairwise_distances_argmin_min(
            self.embs[idx:idx + self.batch_size], self.embs[:idx])
        self._dists_new = pairwise.euclidean_distances(
            self.embs[idx:idx + self.batch_size], self.embs[idx:idx + self.batch_size])


def get_embs(emb_files, pca, max_embs=None):
    tokens, embs, n_embs = [], [None for _ in emb_files], 0

    for i, fname in tqdm(enumerate(emb_files), total=len(emb_files), desc='Reading embs'):
        data = util.read_data(fname)

        tokens += [token for x in data for token in x['sentence']]
        batch_embs = np.concatenate([
            np.expand_dims(emb, axis=0) for x in data for emb in x['embeddings']
        ], axis=0)
        batch_embs = pca.transform(batch_embs)

        embs[i] = batch_embs
        n_embs += batch_embs.shape[0]
        if max_embs and n_embs > max_embs:
            embs = embs[:i + 1]
            break

    embs = np.concatenate(embs, axis=0)
    assert len(tokens) == embs.shape[0]

    if max_embs:
        embs = embs[:max_embs]
        tokens = tokens[:max_embs]

    return embs, tokens


def get_threshold_dist(embs, tokens):
    # Check distances histogram
    dists = pairwise.euclidean_distances(embs[:10000], embs[:50000])
    np.fill_diagonal(dists, 100)
    min_dists = dists.min(1)
    min_dists.sort()

    # Get threshold such that it keeps singleton ratio
    counts = Counter(tokens[:10000])
    singleton_ratio = sum((x == 1) for x in counts.values()) / len(counts)
    threshold_idx = int(min_dists.shape[0] * singleton_ratio)
    threshold = min_dists[threshold_idx]

    return threshold


def assign_codes(embs, tokens, fname):
    # pylint: disable=no-else-continue

    n_tokens, n_types = len(tokens), np.unique(tokens).shape[0]
    threshold = get_threshold_dist(embs, tokens)
    print('Tokens %d\t Types %d\t Ratio %.4f\t Threshold %.4f' %
          (n_tokens, n_types, n_types / n_tokens, threshold))

    distances = IncrementalDist(embs, 500)
    skip_tokens, curr_code, codes = -1, 0, np.ones(embs.shape[0]).astype(int) * -1

    if util.is_file(fname):
        codes = util.read_data(fname).astype(int)
        codes = codes[:embs.shape[0]]
        curr_code = codes.max() + 1
        skip_tokens = (codes != -1).sum()
        print('Reading partial data. Skipping %d tokens' % (skip_tokens))

    pbar = tqdm(range(embs.shape[0]), desc='Getting codes (0)')
    for i in pbar:
        if i < skip_tokens:
            continue
        elif i == 0:
            codes[i] = curr_code
            curr_code += 1
            continue
        elif (i + 1) % 100000 == 0:
            tqdm.write('Dumping partial data (%d)' % i)
            util.write_data(fname, codes)

        parent, mindist = distances.mindist(i)
        if mindist < threshold:
            codes[i] = codes[parent]
            continue

        codes[i] = curr_code
        curr_code += 1

        pbar.set_description('Getting codes (%d)' % curr_code)

    return codes


def main():
    # pylint: disable=fixme
    args = get_args()

    pca = util.read_data(args.pca_file)

    emb_files = util.get_filenames(args.emb_dir)
    ftemp = args.polyassign_file.replace('.pckl', '--temp.pckl')

    embs, tokens = get_embs(emb_files, pca, max_embs=1000000)
    codes = assign_codes(embs, tokens, ftemp)

    counts = Counter(codes)
    count_seq = [counts[i] for i in range(len(counts))]

    print('Tokens %d\t Types %d\t Ratio %.4f' % (len(codes), len(counts), len(counts) / len(codes)))
    print('Frequency vs Order:', stats.spearmanr(count_seq, range(len(counts))))

    util.write_data(args.polyassign_file, codes)


if __name__ == '__main__':
    main()
