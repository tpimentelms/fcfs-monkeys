import os
import pathlib
import random
import io
import csv
import pickle
import numpy as np
import torch


def get_folds():
    # define training set, development set and test set respectively
    # folds range from 0 to 9
    return [list(range(8)), [8], [9]]


def config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_csv(filename, results):
    with io.open(filename, 'a', encoding='utf8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(results)


def overwrite_csv(filename, results):
    with io.open(filename, 'w', encoding='utf8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(results)


def write_data(filename, embeddings):
    with open(filename, "wb") as f:
        pickle.dump(embeddings, f)


def read_data(filename):
    with open(filename, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def read_data_if_exists(filename):
    try:
        return read_data(filename)
    except FileNotFoundError:
        return {}


def remove_if_exists(fname):
    try:
        os.remove(fname)
    except FileNotFoundError:
        pass


def get_filenames(filepath):
    filenames = [os.path.join(filepath, f)
                 for f in os.listdir(filepath)
                 if os.path.isfile(os.path.join(filepath, f))]
    return sorted(filenames)


def get_dirs(filepath):
    filenames = [os.path.join(filepath, f)
                 for f in os.listdir(filepath)
                 if os.path.isdir(os.path.join(filepath, f))]
    return sorted(filenames)


def is_file(fname):
    return os.path.exists(fname)


def mkdir(folder):
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True)


def permutation_test(array, n_permuts=500000):
    mean = abs(array.mean())

    permuts = np.random.randint(0, 2, size=(n_permuts, array.shape[0])) * 2 - 1
    permut_means = np.abs((array * permuts).mean(-1))
    n_larger = (permut_means > mean).sum()

    return (n_larger + 1) / n_permuts
