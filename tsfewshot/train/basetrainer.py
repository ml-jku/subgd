import logging
from abc import ABC, abstractmethod
from shutil import copyfile

import pandas as pd
import torch

from tsfewshot import metrics, utils
from tsfewshot.config import Config
from tsfewshot.logger import Logger
from tsfewshot.loss import get_loss
from tsfewshot.models import get_model
from tsfewshot.test import get_tester

LOGGER = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base class to train a model.

    Subclasses of this class implement specific training strategies.

    Parameters
    ----------
    cfg : Config
        Run configuration
    is_finetune : bool, optional
        Finetune an existing model on a full dataset. The difference to eval with finetune: true is that
        here we don't finetune in a few-shot setting but in normal supervised training with a train and validation set.
    """

    def __init__(self, cfg: Config, is_finetune: bool = False):
        super().__init__()

        utils.set_seed(cfg.seed)

        cfg = utils.setup_run_dir(cfg)
        utils.setup_logging(str(cfg.run_dir / 'output.log'))

        self._tb_logger = Logger(cfg)

        cfg.log_config()
        cfg.dump_config(cfg.run_dir / 'config.yml')

        self._cfg = cfg
        self._device = cfg.device
        self._is_finetune = is_finetune

        self._n_epochs = self._cfg.epochs if not self._is_finetune else self._cfg.finetune_epochs
        if self._n_epochs < 1:
            raise ValueError('epochs should be > 0')

        self._datasets = None
        self._loaders = None
        self._optimizer = None
        self._scheduler = None

        self.noise_sampler_y = None
        if cfg.target_noise_std > 0:
            self.noise_sampler_y = torch.distributions.Normal(loc=0, scale=cfg.target_noise_std)

        # in finetune mode, we load the model with is_test = True, so we'll get a model with only one input/output
        # layer, regardless of layer_per_dataset. During load_model, an eventual layer_per_dataset-model will be
        # loaded and averaged into the single set of weights.
        self._model = get_model(cfg, is_test=self._is_finetune).to(self._device)
        self._loss = get_loss(cfg)
        self._clip_gradient_norm = cfg.clip_gradient_norm
        if is_finetune:
            LOGGER.info(f'Loading model from checkpoint {cfg.checkpoint_path}.')
            self._model, _ = utils.load_model(cfg,
                                              self._model if self._model.is_pytorch else None,
                                              model_path=cfg.checkpoint_path)
            # copy scaler to new run dir
            copyfile(cfg.base_run_dir / 'scaler.p', cfg.run_dir / 'scaler.p')  # type: ignore

        if self._model.is_pytorch:
            # reduce the per-dataset layer from training into one averaged layer
            if is_finetune and cfg.layer_per_dataset is not None and cfg.layer_per_dataset_eval:
                # would work for mixedrotation/singlerotation, but for output we'd need to implement choosing the
                # right set of weights.
                raise NotImplementedError('Per-dataset layer during finetuning not implemented.')
            optimizer = cfg.optimizer['train' if not is_finetune else 'finetune']
            learning_rate = cfg.learning_rate if not is_finetune else cfg.finetune_lr
            self._optimizer, self._scheduler = utils.get_optimizer_and_scheduler(optimizer,
                                                                                 self._model,
                                                                                 learning_rate[0],
                                                                                 learning_rate[1],
                                                                                 learning_rate[2],
                                                                                 weight_decay=cfg.weight_decay)

        self._load_data()
        if len(self._datasets) == 0:  # type: ignore
            raise ValueError('No training data.')

        # mode 'finetune' trains on val and evaluates on test. The reason is mostly convenience: this way,
        # we can leave cfg.train_dataset untouched, so layer_per_dataset can still use it to figure out the right
        # size of per-dataset layers.
        self._val_tester = get_tester(cfg, split='val' if not is_finetune else 'test', tb_logger=self._tb_logger)

    @abstractmethod
    def _load_data(self):
        pass

    def train(self) -> int:
        """Train for n_epochs epochs using early-stopping.

        Returns
        -------
        int
            Number of the best epoch (starting to count with 1).
        """
        self._tb_logger.start_tb()

        best_epoch = 0
        lower_is_better = metrics.lower_is_better(self._cfg.metric[0])
        best_val = float('inf') if lower_is_better else -float('inf')
        for epoch in range(1, self._n_epochs + 1):
            self._model.train()
            self._tb_logger.train()
            self._train_epoch(epoch)
            self._tb_logger.log_epoch()

            if self._scheduler is not None:
                self._scheduler.step()

            # we'll repeat the saving code below in case save_every < 0, but we do it here already to make sure
            # we saved the model even if validation fails.
            saved_model = False
            if (self._cfg.save_every > 0 and epoch % self._cfg.save_every == 0) or epoch == self._n_epochs:
                model_path = self._cfg.run_dir / f'model_epoch{str(epoch).zfill(3)}.p'
                utils.save_model(self._model, model_path)
                saved_model = True

            if epoch % self._cfg.eval_every == 0 \
                    and (self._cfg.val_n_random_datasets is None
                         or (isinstance(self._cfg.val_n_random_datasets, int) and self._cfg.val_n_random_datasets > 0)
                         or (isinstance(self._cfg.val_n_random_datasets, list)
                             and len(self._cfg.val_n_random_datasets) > 0)):
                self._model.eval()
                val_metrics, global_val_metric = self._val_tester.evaluate(trained_model=self._model,  # type: ignore
                                                                           epoch=epoch)
                if not self._cfg.silent:
                    with pd.option_context('display.max_rows', None,
                                           'display.max_columns', None,
                                           'display.width', 1000,
                                           'display.max_colwidth', 300):  # type: ignore
                        LOGGER.info(f'\n{val_metrics}')
                        if global_val_metric is not None:
                            LOGGER.info(f'Global metric:\n{global_val_metric}')

                # get mean/median metric across all datasets and all targets
                if self._cfg.metric_aggregation == 'mean':
                    global_val_metric: float = val_metrics[self._cfg.metric[0]].mean()  # type: ignore
                elif self._cfg.metric_aggregation == 'median':
                    global_val_metric: float = val_metrics[self._cfg.metric[0]].median()  # type: ignore
                elif self._cfg.metric_aggregation == 'global':
                    if global_val_metric is None:
                        raise ValueError('Cannot use global metric for early stopping since it was not calculated.')
                    global_val_metric: float = global_val_metric[self._cfg.metric[0]]  # type: ignore
                else:
                    raise ValueError(f'Unknown aggregation {self._cfg.metric_aggregation}.')
                if pd.isna(global_val_metric):
                    LOGGER.warning(f'Val metrics are all-NaN in epoch {epoch}. Stopping.')
                    break
                if (lower_is_better and global_val_metric < best_val) or \
                        (not lower_is_better and global_val_metric > best_val):
                    best_val = global_val_metric
                    best_epoch = epoch
                    if self._cfg.save_every < 0 and not saved_model:
                        model_path = self._cfg.run_dir / f'model_epoch{str(epoch).zfill(3)}.p'
                        utils.save_model(self._model, model_path)
                if self._cfg.early_stopping_patience is not None:
                    if ((lower_is_better and global_val_metric >= best_val)
                            or (not lower_is_better and global_val_metric <= best_val)) \
                            and epoch > best_epoch + self._cfg.early_stopping_patience:
                        LOGGER.info('Early stopping patience exhausted. '
                                    f'Best val metric {best_val} in epoch {best_epoch}.')
                        break

        if best_epoch > 0:
            with open(self._cfg.run_dir / 'best_epoch.txt', 'w') as fp:
                fp.write(str(best_epoch))

        self._tb_logger.stop_tb()
        return best_epoch

    @abstractmethod
    def _train_epoch(self, epoch: int):
        pass
