import copy
import logging
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from tsfewshot.config import Config
from tsfewshot.models.basemodel import BasePytorchModel
from tsfewshot.train.episodictrainer import EpisodicTrainer
from tsfewshot.utils import get_optimizer_and_scheduler

LOGGER = logging.getLogger(__name__)


class ReptileTrainer(EpisodicTrainer):
    """Class to train a PyTorch model in a Reptile setup.

    Parameters
    ----------
    cfg : Config
        Run configuration
    is_finetune : bool, optional
        Finetune an existing model on a full dataset. The difference to eval with finetune: true is that
        here we don't finetune in a few-shot setting but in normal supervised training with a train and validation set.
    """

    def __init__(self, cfg: Config, is_finetune: bool = False):
        super().__init__(cfg, is_finetune=is_finetune)

        if cfg.optimizer['finetune'] != 'sgd':
            LOGGER.warning('Reptile inner loop will use SGD optimizer.')

        self._n_inner_iter = cfg.maml_n_inner_iter

        self._outer_optimizer = cfg.optimizer['train' if not self._is_finetune else 'finetune']
        self._outer_learning_rate = cfg.learning_rate[0] if not self._is_finetune else cfg.finetune_lr[0]
        self._weight_decay = cfg.weight_decay
        if (not self._is_finetune and cfg.learning_rate[1] is not None) \
                or (self._is_finetune and cfg.finetune_lr[1] is not None):
            raise NotImplementedError('Reptile is not implemented with LR scheduling.')

        self._inner_learning_rate = cfg.maml_inner_lr
        if not isinstance(self._inner_learning_rate, float):
            raise ValueError('Reptile inner lr must be a float')

    def _train_epoch(self, epoch: int):

        episode_iter = iter(self._loaders)
        weight_changes = {name: torch.tensor(0.0).to(self._device)  # pylint: disable=not-callable
                          for name, _ in self._model.named_parameters()}

        for _ in range(self._batch_size):
            # sample two random sets query examples and support examples
            support_set, query_set = next(episode_iter)

            old_params = copy.deepcopy(dict(self._model.named_parameters()))
            initial_states = copy.deepcopy(self._model.state_dict())

            if isinstance(support_set['x'], torch.Tensor):
                support_x = support_set['x'].to(self._device)
            else:
                support_x = support_set['x']
            support_y = support_set['y'].to(self._device)
            loss_kwargs = {}
            if 'std' in support_set.keys():
                loss_kwargs['std'] = support_set['std'].to(self._device)

            if self.noise_sampler_y is not None:
                support_y = support_y + self.noise_sampler_y.sample(support_y.shape).to(support_y)

            for _ in range(self._n_inner_iter):
                self._model.zero_grad()
                support_pred = self._model(support_x)
                support_loss = self._loss(support_pred, support_y, **loss_kwargs)
                support_loss.backward()

                if self._clip_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(),  # type: ignore
                                                   self._clip_gradient_norm)

                with torch.no_grad():
                    for name, param in self._model.named_parameters():
                        update = self._inner_learning_rate * param.grad
                        param.sub_(update)

                        if torch.isnan(param).any():
                            raise ValueError(f'Param {name} has NaNs.')

            self._tb_logger.log_step(loss=support_loss.item())  # type: ignore

            weights_after = self._model.named_parameters()
            for name, param in weights_after:
                weight_changes[name] = weight_changes[name] + (old_params[name] - param)
            self._model.load_state_dict(initial_states)

        # Reptile meta-update. Need to create a new optimizer each time to make it update the right parameter copy.
        # Note/TODO: this resets the optimizer state every time (not relevant for SGD, but for Adam).
        # Empirically seemed to work better this way, but probably that's more a question of
        # doing a proper HP search.
        optimizer, _ = get_optimizer_and_scheduler(self._outer_optimizer,
                                                   self._model,
                                                   self._outer_learning_rate,
                                                   weight_decay=self._weight_decay)
        with torch.no_grad():
            for name, param in self._model.named_parameters():
                param.grad.data = weight_changes[name] / self._batch_size
        optimizer.step()
