import os
import sys
import math
from collections import Counter
from tqdm import tqdm
import numpy as np
import pandas as pd
from ordered_set import OrderedSet

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h02_polysemy.get_polyassign_code import load_polysemy_codes
from util import util
from util.argparser import get_argparser, parse_args


def get_args():
    argparser = get_argparser()
    argparser.add_argument('--polyassign-file', type=str, required=True)
    argparser.add_argument('--pca-file', type=str, required=True)
    argparser.add_argument('--emb-dir', type=str, required=True)
    argparser.add_argument('--polyassign-polysemy-file', type=str, required=True)
    argparser.add_argument('--natural-polysemy-file', type=str, required=False)

    args = parse_args(argparser)
    util.config(args.seed)
    return args


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


def get_gaussian_entropy(cov, base=2):
    # entropy = multivariate_normal.entropy(cov=cov)
    _, logdet = np.linalg.slogdet(2 * np.pi * np.e * cov)
    return 0.5 * logdet / math.log(base)


def get_gaussian_entropy_from_variance(variance, base=2):
    cov = np.diagflat(variance)
    return get_gaussian_entropy(cov, base=base)


def get_polysemy_entropy(token_embs, min_count=2):
    if token_embs.shape[0] < min_count:
        return -float('inf'), -float('inf')

    var = np.var(token_embs, axis=0, ddof=1)
    entropy_var = get_gaussian_entropy_from_variance(var, base=2)
    assert var.shape[0] == 100

    cov = np.cov(token_embs, rowvar=False)
    entropy_cov = get_gaussian_entropy(cov, base=2)
    assert (cov.shape[0] == 100) and (cov.shape[1] == 100)

    return entropy_var, entropy_cov


def get_polysemy(codes, embs, wordforms):
    counts = Counter(codes)

    polysemy_info = []
    for i in tqdm(range(codes.max()), desc='Getting polysemy estimates'):
        wordform = wordforms[i]

        token_embs = embs[codes == i]
        var, cov = get_polysemy_entropy(token_embs)

        polysemy_info += [{
            'idx': i,
            'wordform': wordform,
            'poly_var': var,
            'poly_cov': cov,
            'length': len(wordform),
            'count': counts[i],
        }]

    return polysemy_info


def load_embeddings(args, max_embs=None):
    pca = util.read_data(args.pca_file)
    emb_files = util.get_filenames(args.emb_dir)
    embs, tokens = get_embs(emb_files, pca, max_embs=max_embs)

    return embs, tokens


def save_results(fname, results):
    df = pd.DataFrame(results)
    df.to_csv(fname, sep='\t', index=False)


def main():
    # pylint: disable=too-many-locals
    args = get_args()

    poly_codes = load_polysemy_codes(args.polyassign_file)
    poly_types = [str(i) for i in range(len(set(poly_codes)))]
    embs, tokens = load_embeddings(args, max_embs=poly_codes.shape[0])

    types = list(OrderedSet(tokens))
    token_map = {token: i for i, token in enumerate(types)}
    natural_codes = np.array([token_map[token] for token in tokens])

    polyassign_polysemy_info = get_polysemy(
        poly_codes, embs, poly_types)
    save_results(args.polyassign_polysemy_file, polyassign_polysemy_info)

    natural_polysemy_info = get_polysemy(
        natural_codes, embs, types)
    save_results(args.natural_polysemy_file, natural_polysemy_info)


if __name__ == '__main__':
    main()
