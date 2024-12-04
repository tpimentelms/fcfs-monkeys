import torch
from torch.utils.data import DataLoader

from util import constants
from util import util
from .types import TypeDataset


def generate_batch(batch):
    tensor = batch[0][0]
    batch_size = len(batch)
    max_length = max(len(entry[0]) for entry in batch) - 1  # Does not need to predict SOS

    x = tensor.new_zeros(batch_size, max_length)
    y = tensor.new_zeros(batch_size, max_length)

    for i, item in enumerate(batch):
        word = item[0]
        word_len = len(word) - 1  # Does not need to predict SOS
        x[i, :word_len] = word[:-1]
        y[i, :word_len] = word[1:]

    x, y = x.to(device=constants.device), y.to(device=constants.device)
    weights = torch.cat([b[1] for b in batch]).to(device=constants.device)
    indices = torch.LongTensor([b[2] for b in batch]).to(device=constants.device)
    tokens = [b[3] for b in batch]
    return x, y, weights, indices, tokens


def load_data(fname):
    return util.read_data(fname)


def get_alphabet(data):
    alphabet = data[1]
    return alphabet


def get_data_loader(
        data, folds, batch_size, shuffle):
    if not folds:
        print('Warning: empty fold passed to data loader.')
        return None
    trainset = TypeDataset(data, folds)
    return DataLoader(trainset, batch_size=batch_size,
                      shuffle=shuffle, collate_fn=generate_batch)


def get_data_loaders(
        fname, folds, batch_size=128, eval_batch_size=128):
    data = load_data(fname)
    alphabet = get_alphabet(data)

    trainloader = get_data_loader(
        data, folds[0], batch_size=batch_size, shuffle=True)
    devloader = get_data_loader(
        data, folds[1], batch_size=eval_batch_size, shuffle=True)
    testloader = get_data_loader(
        data, folds[2], batch_size=eval_batch_size, shuffle=False)
    return trainloader, devloader, testloader, alphabet
