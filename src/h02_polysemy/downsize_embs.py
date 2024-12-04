import os
import sys
from tqdm import tqdm
import numpy as np
from numpy.random import default_rng
from sklearn.decomposition import PCA

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import util
from util.argparser import get_argparser, parse_args


def get_args():
    argparser = get_argparser()
    argparser.add_argument('--emb-dir', type=str, required=True)
    argparser.add_argument('--pca-file', type=str, required=True)
    argparser.add_argument('--language', type=str, required=True)
    argparser.add_argument('--n-samples', type=int, default=200000)

    args = parse_args(argparser)
    util.config(args.seed)
    return args


def get_n_embs(emb_files):
    n_sentences, n_embs = 0, 0
    pbar = tqdm(emb_files, desc='Counting sentences (0)')
    for fname in pbar:
        data = util.read_data(fname)

        n_sentences += len(data)
        n_embs += sum(len(x['sentence']) for x in data)
        pbar.set_description("Counting sentences (%d / %d)" % (n_sentences, n_embs))

    return n_sentences, n_embs


def sample_embs(emb_files, n_embs, n_samples):
    rng = default_rng()
    samples = rng.choice(n_embs, size=n_samples, replace=False)
    embs = []

    for fname in tqdm(emb_files, desc='Extracting embeddings'):
        data = util.read_data(fname)

        batch_embs = np.concatenate([
            np.expand_dims(emb, axis=0) for x in data for emb in x['embeddings']
        ], axis=0)
        batch_size = batch_embs.shape[0]

        batch_samples = samples[samples < batch_size]
        embs += [batch_embs[batch_samples]]

        samples = samples - batch_size
        samples = samples[samples >= 0]

    return np.concatenate(embs, axis=0)


def train_pca(embs):
    pca = PCA(n_components=100)
    pca.fit(embs)
    print('Explained variance:', pca.explained_variance_ratio_.sum())

    return pca


def main():
    args = get_args()

    emb_files = util.get_filenames(args.emb_dir)

    _, n_embs = get_n_embs(emb_files)
    if args.n_samples > n_embs:
        print(('Warning: Too few embeddings (%d) to sample (%d).' +
               'Using all of them') % (n_embs, args.n_samples))
        args.n_samples = n_embs

    embs = sample_embs(emb_files, n_embs, args.n_samples)

    pca = train_pca(embs)
    util.write_data(args.pca_file, pca)


if __name__ == '__main__':
    main()
