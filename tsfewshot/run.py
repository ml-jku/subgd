import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).parent.parent))
from tsfewshot import utils
from tsfewshot.config import Config
from tsfewshot.test import get_tester
from tsfewshot.train import get_trainer

LOGGER = logging.getLogger(__name__)


def _get_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'eval', 'finetune'])
    parser.add_argument('--config-file', type=str)
    parser.add_argument('--run-dir', type=str)
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--epoch', type=int, required=False)
    parser.add_argument('--gpu', type=int, required=False)
    args = vars(parser.parse_args())

    if (args['mode'] == 'train') and (args['config_file'] is None):
        raise ValueError('Need config file to train')
    if args['mode'] == 'train' \
            and (args['epoch'] is not None or args['run_dir'] is not None):
        raise ValueError('Cannot provide epoch or run-dir for training')

    return args


def _main():
    args = _get_args()

    device = args['gpu']
    if device is not None:
        device = _get_device(device)

    if args['mode'] == 'train':
        cfg = Config(args['config_file'])
        if device is not None:
            cfg.device = torch.device(device)
        _train(cfg)
    elif args['mode'] == 'eval':
        _eval(args['run_dir'], args['split'], config_file=args['config_file'], epoch=args['epoch'], device=device)
    elif args['mode'] == 'finetune':
        _finetune(args['config_file'], device=device)
    else:
        raise ValueError(f'Unknown mode {args["mode"]}')


def _train(cfg: Config):
    trainer = get_trainer(cfg)
    trainer.train()


def _finetune(config_file: str, device: str = None):
    if config_file is None:
        raise ValueError('Need config-file in finetuning.')
    update_config = Config(Path(config_file))
    cfg = Config(update_config.base_run_dir / 'config.yml')
    cfg.device = torch.device('cpu')
    cfg.update({'run_dir': None})  # make sure the finetune run gets a new run dir
    LOGGER.info('Updating train config with provided new config file')
    cfg.update(update_config)
    if device is not None:
        cfg.device = torch.device(device)
    trainer = get_trainer(cfg, is_finetune=True)
    trainer.train()


def _eval(run_dir: str, split: str, config_file: str = None, epoch: int = None, device: str = None):
    if run_dir is None:
        if config_file is None:
            raise ValueError('Need run-dir or config-file in evaluation.')

        # This can be reasonable, e.g., for a linear regression model that will be fit to the support set
        cfg = Config(config_file)

        if not cfg.finetune:
            raise ValueError('Without prior training, finetune must be active')

    else:
        utils.setup_logging(str(Path(run_dir) / 'output.log'))
        cfg = Config(Path(run_dir) / 'config.yml')
        git_hash = utils.get_git_hash()
        if git_hash is not None and git_hash != cfg.commit_hash:
            LOGGER.info(f'New git commit hash: {git_hash} (previously {cfg.commit_hash})')
            cfg.commit_hash = git_hash
        utils.save_git_diff(cfg.run_dir)  # type: ignore
        if config_file is not None:
            update_config = Config(Path(config_file))
            LOGGER.info('Updating train config with provided new config file')
            cfg.update(update_config)
            cfg.log_config()
            cfg.dump_config(cfg.run_dir / 'eval_config.yml', overwrite=True)

    if device is not None:
        cfg.device = torch.device(device)

    tester = get_tester(cfg, split=split, init_model_params=run_dir is not None)

    # if run_dir was None, the tester should not try to load model parameters (because they won't exist)
    metrics, global_metrics = tester.evaluate(epoch=epoch)

    # make sure we print the whole series
    with pd.option_context('display.max_rows', None,
                           'display.max_columns', None,
                           'display.width', 1000,
                           'display.max_colwidth', 300):  # type: ignore
        LOGGER.info(metrics)
        if global_metrics is not None:
            LOGGER.info(global_metrics)


def _get_device(device: int) -> str:
    if device < 0:
        return 'cpu'
    return f'cuda:{device}'


if __name__ == '__main__':
    LOGGER.info(f'Command: python {" ".join(sys.argv)}\n')
    _main()
