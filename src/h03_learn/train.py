import os
import sys
import tqdm
import torch
from torch import optim

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h03_learn.dataset import get_data_loaders
from h03_learn.model import LstmLM
from h03_learn.train_info import TrainInfo
from util.argparser import get_argparser, parse_args, add_all_defaults
from util import util
from util import constants


def get_args():
    argparser = get_argparser()

    add_all_defaults(argparser)
    args = parse_args(argparser)
    args.wait_iterations = args.wait_epochs * args.eval_batches
    return args


def load_model(checkpoints_path):
    return LstmLM.load(checkpoints_path).to(device=constants.device)


def get_model(alphabet, args):
    return LstmLM(
        len(alphabet), args.embedding_size, args.hidden_size,
        nlayers=args.nlayers, dropout=args.dropout,
        ignore_index=alphabet.PAD_IDX) \
        .to(device=constants.device)


def train_batch(x, y, model, optimizer, by_character=False):
    optimizer.zero_grad()
    y_hat = model(x)
    loss = model.get_loss(y_hat, y).sum(-1)
    if by_character:
        word_lengths = (y != 0).sum(-1)
        loss = (loss / word_lengths).mean()
    else:
        loss = loss.mean()
    loss.backward()
    optimizer.step()
    return loss.item()


def train(trainloader, devloader, model, eval_batches, wait_iterations):
    optimizer = optim.AdamW(model.parameters())
    train_info = TrainInfo(wait_iterations, eval_batches)

    while not train_info.finish:
        for x, y, _, _, _ in tqdm.tqdm(trainloader, total=len(trainloader)):
            loss = train_batch(x, y, model, optimizer)
            train_info.new_batch(loss)

            if train_info.eval:
                dev_loss = evaluate(devloader, model, max_samples=10000)

                if train_info.is_best(dev_loss):
                    model.set_best()
                elif train_info.finish:
                    break

                train_info.print_progress(dev_loss)

    model.recover_best()
    return loss, dev_loss


def _evaluate(evalloader, model, max_samples=None):
    dev_loss, n_instances = 0, 0
    for x, y, weights, _, _ in evalloader:
        y_hat = model(x)
        loss = model.get_loss(y_hat, y).sum(-1)
        dev_loss += (loss * weights).sum()
        n_instances += weights.sum()

        if max_samples is not None and n_instances >= max_samples:
            break

    return (dev_loss / n_instances).item()


def evaluate(evalloader, model, max_samples=None):
    model.eval()
    evalloader.dataset.eval()
    with torch.no_grad():
        result = _evaluate(evalloader, model, max_samples=max_samples)
    model.train()
    evalloader.dataset.train()
    return result


def save_results(model, train_loss, dev_loss, train_size, dev_size, results_fname):
    results = [['alphabet_size', 'embedding_size', 'hidden_size', 'nlayers',
                'dropout_p', 'train_loss', 'dev_loss',
                'train_size', 'dev_size']]
    results += [[model.alphabet_size, model.embedding_size, model.hidden_size,
                 model.nlayers, model.dropout_p, train_loss, dev_loss,
                 train_size, dev_size]]
    util.write_csv(results_fname, results)


def save_checkpoints(model, train_loss, dev_loss, train_size, dev_size, checkpoints_path):
    model.save(checkpoints_path)
    results_fname = checkpoints_path + '/graphotactic-results.csv'
    save_results(model, train_loss, dev_loss, train_size, dev_size, results_fname)


def main():
    args = get_args()
    folds = util.get_folds()

    trainloader, devloader, _, alphabet = get_data_loaders(
        args.data_file, folds, args.batch_size, args.eval_batch_size)

    print(f'Train size: {len(trainloader.dataset)} Dev size: {len(devloader.dataset)}')

    model = get_model(alphabet, args)
    train(trainloader, devloader, model, args.eval_batches, args.wait_iterations)

    train_loss = evaluate(trainloader, model)
    dev_loss = evaluate(devloader, model)

    print(f'Final Training loss: {train_loss:.4f} Dev loss: {dev_loss:.4f}')

    save_checkpoints(model, train_loss, dev_loss, len(trainloader.dataset),
                     len(devloader.dataset), args.checkpoints_path)


if __name__ == '__main__':
    main()
