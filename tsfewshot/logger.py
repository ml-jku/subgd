from __future__ import annotations

from collections import defaultdict
from typing import List, Union

import numpy as np
from matplotlib import figure
from torch.utils.tensorboard import SummaryWriter

from tsfewshot.config import Config
from tsfewshot.utils import get_git_hash, save_git_diff


class Logger:
    """Class that logs runs to tensorboard.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        self._mode = 'train'
        self._log_interval = 10
        self._log_dir = cfg.run_dir

        # get git commit hash if folder is a git repository
        git_hash = get_git_hash()
        if git_hash is not None:
            cfg.commit_hash = git_hash
        save_git_diff(cfg.run_dir)  # type: ignore

        self._update = 0
        self._epoch = 0
        self._finetune_epoch = 0
        self._metrics = defaultdict(lambda: defaultdict(list))
        self._losses = defaultdict(lambda: defaultdict(list))
        self._finetune_results = []

        self._use_tb = cfg.log_tensorboard
        self._writer = None

    def train(self):
        """Set logging to training mode. """
        self._mode = 'train'

    def eval(self):
        """Set logging to evaluation mode. """
        self._mode = 'eval'

    def finetune(self):
        """Set logging to finetune mode. """
        self._mode = 'finetune'
        self._finetune_epoch = 0

    def start_tb(self):
        """Start tensorboard logging if it not started already. """
        if self._writer is None and self._use_tb:
            self._writer = SummaryWriter(log_dir=str(self._log_dir))

    def stop_tb(self):
        """Stop tensorboard logging. """
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None

    def log_step(self, dataset: str = '', **kwargs):
        """Log the results of a single step within an epoch.

        Parameters
        ----------
        dataset : str, optional
            Dataset of the step. In train mode, this should be left empty.
        **kwargs, optional
            Key-value pairs of metric names and values.
        """
        if self._mode in ['train', 'finetune']:
            for k, v in kwargs.items():
                self._losses[dataset][k].append(v)
            if self._mode == 'train':
                self._update += 1

                if self._log_interval <= 0 or self._writer is None:
                    return

                if self._update % self._log_interval == 0:
                    for k, v in kwargs.items():
                        self._writer.add_scalar('/'.join([self._mode, k]), v,
                                                self._update)
        else:
            for k, v in kwargs.items():
                self._metrics[dataset][k].append(v)

    def log_epoch(self):
        """"Log the results of the entire training or finetune epoch. """
        if self._mode == 'train':
            self._epoch += 1
            if self._writer is not None:
                for k, v in self._losses[''].items():
                    self._writer.add_scalar('/'.join([self._mode, f'mean_{k.lower()}']), np.nanmean(v),
                                            self._epoch)
                    self._writer.add_scalar('/'.join([self._mode, f'median_{k.lower()}']), np.nanmedian(v),
                                            self._epoch)
            # clear buffer
            self._losses = defaultdict(lambda: defaultdict(list))
        elif self._mode == 'finetune':
            # combine metrics across all datasets so we can later calculate the epoch mean/median across all datasets
            if len(self._finetune_results) <= self._finetune_epoch:
                self._finetune_results.append(defaultdict(list))
            for ds_losses in self._losses.values():
                for k, v in ds_losses.items():
                    self._finetune_results[self._finetune_epoch][k] += v
            self._finetune_epoch += 1
        else:
            return

        self._losses = defaultdict(lambda: defaultdict(list))

    def summarize(self):
        """Log the validation (and possibly finetuning) results to tensorboard. """
        if self._mode == 'train':
            return

        if self._writer is not None:
            # summarize finetuning
            for i, finetune_results in enumerate(self._finetune_results):
                epoch = (self._epoch - 1) * len(self._finetune_results) + i
                for k, v in finetune_results.items():
                    self._writer.add_scalar(f'finetune/mean_{k}', np.nanmean(v), epoch)
                    self._writer.add_scalar(f'finetune/median_{k}', np.nanmedian(v), epoch)
                    if i == len(self._finetune_results) - 1:
                        self._writer.add_scalar(f'finetune/last_{k}', np.nanmean(v), self._epoch)

            # summarize evaluation
            eval_results = defaultdict(list)
            for ds_metrics in self._metrics.values():
                for k, v in ds_metrics.items():
                    eval_results[k].append(v)
            for k, v in eval_results.items():
                self._writer.add_scalar(f'eval/mean_{k}', np.nanmean(v), self._epoch)
                self._writer.add_scalar(f'eval/median_{k}', np.nanmedian(v), self._epoch)

        # clear buffers
        self._finetune_epoch = 0
        self._finetune_results = []
        self._losses = defaultdict(lambda: defaultdict(list))
        self._metrics = defaultdict(lambda: defaultdict(list))

    def log_figure(self, fig: Union[figure.Figure, List[figure.Figure]], tag: str):
        """Log figure to tensorboard and save to png. """
        if self._writer is not None:
            self._writer.add_figure(tag, fig, global_step=self._epoch)
