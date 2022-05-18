import copy
import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from tsfewshot import utils
from tsfewshot.config import Config
from tsfewshot.logger import Logger
from tsfewshot.loss import BaseLoss, get_loss
from tsfewshot.metrics import lower_is_better
from tsfewshot.models import get_model
from tsfewshot.models.basemodel import (BaseModel, BasePytorchModel,
                                        BaseSklearnModel, MetaCurvatureWrapper,
                                        MetaSGDWrapper)
from tsfewshot.models.eulerode import EulerODE
from tsfewshot.models.lstm import LSTM
from tsfewshot.models.manuallstm import ManualLSTM
from tsfewshot.regularization import Regularization
from tsfewshot.test.tester import Tester
from tsfewshot.utils import get_file_path, get_optimizer_and_scheduler

LOGGER = logging.getLogger(__name__)


class FinetuneTester(Tester):
    """Class to test a model with finetuning.

    For each target task, the tester samples cfg.support_size support time series and lets the model generate
    predictions for the remaining samples (queries).

    Parameters
    ----------
    cfg : Config
        Run configuration
    split : {'train', 'val', 'test'}
        Period to evaluate on.
    init_model_params : bool, optional
        If True, will try to load a model from disk if `model` is None. If False, will use the uninitialized model
        that is returned by ``get_model``.
    tb_logger : Logger, optional
        Initialized tensorboard logger.
    """

    def __init__(self, cfg: Config, split: str, init_model_params: bool = True, tb_logger: Logger = None):
        super().__init__(cfg, split, init_model_params=init_model_params, tb_logger=tb_logger)

    def _finetune_hook(self, model: BaseModel,
                       ds_name: str,
                       support_x: torch.Tensor = None,
                       support_y: torch.Tensor = None,
                       support_supplemental: Dict[str, torch.Tensor] = None,
                       query_loader: DataLoader = None) -> BaseModel:
        if support_x is None or support_y is None or support_supplemental is None:
            raise ValueError('Cannot finetune without support set.')

        finetune_model = get_model(self._cfg, is_test=True, is_finetune=True).to(self._device)
        if isinstance(finetune_model, BaseSklearnModel):
            return self._finetune_sklearn(model, finetune_model, support_x=support_x, support_y=support_y)

        return self._finetune_pytorch(model, finetune_model,  # type: ignore
                                      support_x=support_x,
                                      support_y=support_y,
                                      support_supplemental=support_supplemental,
                                      query_loader=query_loader,
                                      ds_name=ds_name)

    def _finetune_sklearn(self,
                          original_model: BaseModel,
                          finetune_model: BaseSklearnModel,
                          support_x: torch.Tensor,
                          support_y: torch.Tensor) -> BaseSklearnModel:
        finetune_model = copy.deepcopy(original_model)  # type: ignore

        predict_last_n = self._cfg.predict_last_n
        if predict_last_n is not None:
            support_x = support_x[:, -predict_last_n:]
            support_y = support_y[:, -predict_last_n:]
        finetune_model.fit(support_x, support_y)

        return finetune_model

    def _finetune_pytorch(self,
                          original_model: BasePytorchModel,
                          finetune_model: BasePytorchModel,
                          support_x: torch.Tensor,
                          support_y: torch.Tensor,
                          support_supplemental: Dict[str, torch.Tensor],
                          ds_name: str,
                          query_loader: DataLoader = None) -> BasePytorchModel:

        # directly deepcopying the model prints performance warnings, so we copy the state into a new model
        if isinstance(finetune_model, ManualLSTM) and isinstance(original_model, (LSTM, ManualLSTM)):
            # don't copy the head if the new model should use other gates/states than the trained model
            finetune_model.copy_weights(original_model, copy_head=(
                self._cfg.lstm_head_inputs['train'] == self._cfg.lstm_head_inputs['finetune']),
                copy_input_layer=True)
        else:
            # average the head into a single set of weights, then replicate for the number of classes
            if self._cfg.classification_n_classes['train'] != self._cfg.classification_n_classes['finetune']:
                finetune_model = utils.update_classification_head(self._cfg, finetune_model)

            finetune_model.load_state_dict(copy.deepcopy(original_model.state_dict()))

        finetune_model = self._freeze_modules(finetune_model)
        finetune_model.train()

        optimizer, scheduler = self._get_optimizer(finetune_model)

        loss_fn = get_loss(self._cfg)
        reg_fn = Regularization(self._cfg, original_model)

        self._pre_finetune_hook(finetune_model, query_loader=query_loader, dataset=ds_name)

        metrics = {}
        best_epoch = 0
        lower_better = lower_is_better(self._cfg.metric[0])
        best_val = float('inf') if lower_better else -float('inf')
        for i in range(self._cfg.finetune_epochs):
            finetune_model, is_nan = self._finetune_epoch(ds_name,
                                                          finetune_model,
                                                          optimizer,
                                                          loss_fn,
                                                          reg_fn,
                                                          support_x,
                                                          support_y)
            if scheduler is not None:
                scheduler.step()
            if is_nan:
                LOGGER.warning(f'Loss/regularization was NaN/inf in finetuning epoch {i+1}. Stopped finetuning.')
                break

            if self._tb_logger is not None:
                self._tb_logger.log_epoch()

            ft_eval_every = self._cfg.finetune_eval_every
            if ft_eval_every is not None and \
                ((isinstance(ft_eval_every, int) and i % ft_eval_every == 0)
                    or (isinstance(ft_eval_every, list) and i + 1 in ft_eval_every)):
                if query_loader is None:
                    raise ValueError('Cannot calculate metrics without query set.')

                # get rescaled absolute predictions/targets
                finetune_model.eval()
                save_file = None
                if self._cfg.save_predictions:
                    predictions_dir = self._cfg.run_dir / 'predictions/finetune'  # type: ignore
                    predictions_dir.mkdir(parents=True, exist_ok=True)
                    save_file = get_file_path(predictions_dir,
                                              split=self._split,
                                              ds_name=ds_name,
                                              epoch=str(i),
                                              ext='nc')
                queries_y_abs, predictions_abs, query_sample_ids, _ = self._get_predictions(finetune_model,
                                                                                            ds_name=ds_name,
                                                                                            query_loader=query_loader,
                                                                                            save_file=save_file)
                metrics[i] = self._get_metrics(ds_name,
                                               queries_y_abs,
                                               predictions_abs,
                                               query_sample_ids)[self._cfg.metric[0]]
                finetune_model.train()
                if (lower_better and metrics[i] < best_val) or (not lower_better and metrics[i] > best_val):
                    best_val = metrics[i]
                    best_epoch = i
                if self._cfg.finetune_early_stopping_patience is not None:
                    if ((lower_better and metrics[i] >= best_val)
                            or (not lower_better and metrics[i] <= best_val)) \
                            and i > best_epoch + self._cfg.finetune_early_stopping_patience:
                        LOGGER.info('Early stopping patience exhausted. '
                                    f'Best val metric {best_val} in epoch {best_epoch}.')
                        break

        self._post_finetune_hook(ds_name)

        if len(metrics) > 0:
            metrics_path = self._cfg.run_dir / 'results' / 'finetune'
            metrics_path.mkdir(parents=True, exist_ok=True)
            pd.Series(metrics).to_csv(metrics_path / f'{ds_name.replace("/", "")}.csv')

        return finetune_model

    def _finetune_epoch(self, ds_name: str,
                        finetune_model: BasePytorchModel,
                        optimizer: optim.Optimizer,
                        loss_fn: BaseLoss,
                        reg_fn: Regularization,
                        support_x: torch.Tensor,
                        support_y: torch.Tensor) -> Tuple[BasePytorchModel, bool]:
        # use support shape, not cfg.support_size for MAML/MetaSGD/Reptile, because in N-way settings
        # there will be N * support_size support samples.
        batch_size = self._cfg.batch_size if self._cfg.training_setup not in ['maml', 'metasgd', 'reptile'] \
            else support_y.shape[0]
        n_batches = math.ceil(support_y.shape[0] / batch_size)
        shuffled_indices = np.random.permutation(len(support_y))
        if isinstance(finetune_model, EulerODE)  \
                or (hasattr(finetune_model, 'model') and isinstance(finetune_model.model, EulerODE)):
            assert n_batches == 1, "For eulerode models support set must only have one batch."
            # EulerODE expects dict as input -> support_y is a dict
            shuffled_support_x = [support_x]
        else:
            shuffled_support_x = support_x[shuffled_indices]
        shuffled_support_y = support_y[shuffled_indices]
        for j in range(n_batches):
            batch_slice = slice(j * batch_size, (j + 1) * batch_size)
            batch_x = shuffled_support_x[batch_slice]
            batch_y = shuffled_support_y[batch_slice]
            loss_kwargs = {}
            if self._cfg.loss_fn == 'nse':
                key = ds_name
                if key not in self._datasets:
                    # if val_n_random_datasets is set, the dataset can have a suffix "_\d+" which we have to remove
                    # for the std lookup.
                    key, suffix = ds_name.rsplit('_', maxsplit=1)
                    if not all(s in '0123456789' for s in suffix):
                        raise ValueError(f'std missing for dataset {ds_name}')
                loss_kwargs['std'] = self._datasets[key].stds[key].repeat(batch_y.shape[0]).to(self._device)
            optimizer.zero_grad()

            if isinstance(finetune_model, EulerODE)  \
                    or (hasattr(finetune_model, 'model') and isinstance(finetune_model.model, EulerODE)):
                # batch_x is a list with one item -> access the item
                prediction = finetune_model(batch_x[0])
            else:
                prediction = finetune_model(batch_x)

            loss = loss_fn(prediction, batch_y, **loss_kwargs)
            reg = reg_fn(finetune_model)
            (loss + reg).backward()

            if self._cfg.clip_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(finetune_model.parameters(),  # type: ignore
                                               self._cfg.clip_gradient_norm)

            if self._tb_logger is not None:
                self._tb_logger.log_step(dataset=ds_name, loss=loss.item(), regularization=reg.item())

            self._pre_step_hook(finetune_model)

            optimizer.step()

            self._post_step_hook(finetune_model)

            # stop after the optimizer step so the following evaluation will realize the model being NaN.
            if torch.isnan(loss + reg) or torch.isinf(loss + reg):
                return finetune_model, True

        return finetune_model, False

    def _freeze_modules(self, model: BasePytorchModel) -> BasePytorchModel:
        if self._cfg.finetune_modules is None:
            return model
        if len(self._cfg.finetune_modules) == 0:
            raise ValueError('finetune_modules is empty. Cannot finetune with all weights frozen.')

        # freeze all model weights
        for param in model.parameters():
            param.requires_grad = False

        # unfreeze parameters specified in config as tuneable parameters
        if isinstance(self._cfg.finetune_modules, list):
            for module_part in self._cfg.finetune_modules:
                module = getattr(model, module_part)
                if isinstance(module, nn.Parameter):
                    module.requires_grad = True
                else:
                    for param in module.parameters():
                        param.requires_grad = True
        else:
            # if it was no list, it has to be a dictionary
            for module_group, module_parts in self._cfg.finetune_modules.items():
                if isinstance(module_parts, str):
                    module_parts = [module_parts]
                for module_part in module_parts:
                    module = getattr(model, module_group)
                    submodule = getattr(module, module_part)
                    if isinstance(submodule, nn.Parameter):
                        submodule.requires_grad = True
                    else:
                        for param in submodule.parameters():
                            param.requires_grad = True

        return model

    def _get_optimizer(self, finetune_model: BasePytorchModel) \
            -> Tuple[optim.Optimizer, Optional[optim.lr_scheduler.MultiStepLR]]:
        return get_optimizer_and_scheduler(self._cfg.optimizer['finetune'],
                                           finetune_model,
                                           self._cfg.finetune_lr[0],
                                           self._cfg.finetune_lr[1],
                                           self._cfg.finetune_lr[2],
                                           weight_decay=self._cfg.weight_decay)

    def _pre_finetune_hook(self, finetune_model: BasePytorchModel, dataset: str, query_loader: DataLoader = None):
        pass

    def _pre_step_hook(self, finetune_model: BasePytorchModel):
        if isinstance(finetune_model, MetaSGDWrapper):
            with torch.no_grad():
                # apply per-parameter learning rate of MetaSGD
                for param, lr in zip(finetune_model.model.parameters(), finetune_model.learning_rates):
                    if param.grad is not None:
                        param.grad = lr * param.grad
        elif isinstance(finetune_model, MetaCurvatureWrapper):
            with torch.no_grad():
                # apply preconditioning of Meta-Curvature
                for param, m_in, m_out in zip(finetune_model.model.parameters(),
                                              finetune_model.m_in,
                                              finetune_model.m_out):
                    if param.grad is not None:
                        param.grad = torch.mm(m_out, torch.mm(param.grad, m_in)) if param.ndim == 2 \
                            else m_in * param.grad

    def _post_step_hook(self, finetune_model: BasePytorchModel):
        pass

    def _post_finetune_hook(self, ds_name: str):
        pass
