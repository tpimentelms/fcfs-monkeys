import os
import sys
from tqdm import tqdm
import torch

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from h03_learn.dataset import get_data_loaders
from h03_learn.train import load_model
from util.argparser import get_argparser, parse_args, add_data_args
from util import util


def get_args():
    argparser = get_argparser()
    # Models
    argparser.add_argument('--eval-path', type=str, required=True)
    add_data_args(argparser)
    return parse_args(argparser)


def _get_logprobs_dataset(evalloader, model, logprobs):
    for x, y, _, _, tokens in tqdm(evalloader, desc='Getting logprobs', total=len(evalloader)):
        y_hat = model(x)
        loss = model.get_loss(y_hat, y).sum(-1)

        assert len(tokens) == loss.shape[0]
        for token, logprob in zip(tokens, loss):
            logprobs[token] = logprob.item()


def get_logprobs_dataset(evalloader, model, logprobs):
    model.eval()
    evalloader.dataset.eval()
    with torch.no_grad():
        _get_logprobs_dataset(evalloader, model, logprobs)


def get_logprobs(model_path, trainloader, devloader, testloader):
    model = load_model(model_path)

    logprobs = {}
    get_logprobs_dataset(trainloader, model, logprobs)
    get_logprobs_dataset(devloader, model, logprobs)
    get_logprobs_dataset(testloader, model, logprobs)

    avg_logprob = sum(logprobs.values()) / len(logprobs)
    print(f'Avg type loss: {avg_logprob:.4f}')

    return logprobs


def main():
    args = get_args()
    folds = util.get_folds()

    trainloader, devloader, testloader, _ = \
        get_data_loaders(args.data_file, folds,
                                    args.eval_batch_size, args.eval_batch_size)

    print(f'Train size: {len(trainloader.dataset)} ' +
          f'Dev size: {len(devloader.dataset)} ' +
          f'Test size: {len(testloader.dataset)}')

    logprobs = get_logprobs(args.eval_path, trainloader, devloader, testloader)

    results_file = os.path.join(args.eval_path, 'graphotactic-logprobs.pckl')
    util.write_data(results_file, logprobs)


if __name__ == '__main__':
    main()
