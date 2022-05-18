import logging
import sys
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tsfewshot.config import Config
from tsfewshot.data import get_dataset
from tsfewshot.data.basedataset import BaseDataset
from tsfewshot.train.basetrainer import BaseTrainer

LOGGER = logging.getLogger(__name__)


class SupervisedTrainer(BaseTrainer):
    """Class to train a model in a supervised setup.

    This class will train a model in the normal supervised setup. For PyTorch models, this means mini-batched training.
    For sklearn models, the training samples are all passed to the model at once.
    The specific task that a sample belongs to is ignored.

    Parameters
    ----------
    cfg : Config
        Run configuration
    is_finetune : bool, optional
        Finetune an existing model on a full dataset. The difference to eval with finetune: true is that
        here we don't finetune in a few-shot setting but in normal supervised training with a train and validation set.
    """

    def __init__(self, cfg: Config, is_finetune: bool = False):
        self._train_indices: np.ndarray

        super().__init__(cfg, is_finetune=is_finetune)

        self._store_path = cfg.store_training_path
        if self._store_path:
            if not self._model.is_pytorch:
                raise ValueError('Can only store training path for PyTorch models.')
            if len(list(self._cfg.run_dir.glob('train_path_epoch*.p'))) > 0:  # type: ignore
                raise ValueError('train_path file exists.')
            LOGGER.warning('Storing the full training path to disk.')
        # initial parameters
        self._training_path = [{name: p.detach().cpu() for name, p in self._model.named_parameters()}]

        if self._cfg.train_val_split is not None:
            train_ds = self._cfg.train_datasets if not self._is_finetune else self._cfg.val_datasets
            val_ds = self._cfg.val_datasets if not self._is_finetune else self._cfg.test_datasets
            if train_ds != val_ds:
                raise ValueError('Can only create supervised train/val split if train and val datasets are identical.')

            self._val_tester.subset_test_datasets(self._datasets.train_indices)   # type: ignore

    def _load_data(self):
        scaler = None
        if self._is_finetune:
            scaler = BaseDataset.load_scaler_from_disk(self._cfg)

        # is_train must be False on finetuning so that we get the averaged layer_per_dataset layer.
        # mode 'finetune' trains on val and evaluates on test. The reason is mostly convenience: this way,
        # we can leave cfg.train_dataset untouched, so layer_per_dataset can still use it to figure out the right
        # size of per-dataset layers.
        self._datasets = get_dataset(self._cfg, split='train' if not self._is_finetune else 'val',
                                     is_train=not self._is_finetune,
                                     train_scaler=scaler,
                                     train_val_split=self._cfg.train_val_split,
                                     silent=self._cfg.silent)
        self._create_loader()

    def _create_loader(self):
        self._loaders = DataLoader(self._datasets,
                                   batch_size=self._cfg.batch_size if self._model.is_pytorch else len(self._datasets),
                                   shuffle=self._model.is_pytorch,
                                   num_workers=self._cfg.num_workers,
                                   pin_memory=True)

    def _train_epoch(self, epoch: int):
        if self._model.is_pytorch:
            self._train_epoch_pytorch(epoch)
        else:
            if epoch > 1:
                raise ValueError('It does not make sense to train an sklearn model for multiple epochs.')
            self._train_sklearn()

    def _train_epoch_pytorch(self, epoch: int):
        pbar = tqdm(self._loaders, file=sys.stdout, disable=self._cfg.silent)
        pbar.set_description(f'Epoch {epoch}')
        total_loss = 0.0
        for batch in pbar:
            self._optimizer.zero_grad()

            if isinstance(batch['x'], torch.Tensor):
                x = batch['x'].to(self._device)
            else:
                x = batch['x']
            y = batch['y'].to(self._device)

            if self.noise_sampler_y is not None:
                y = y + self.noise_sampler_y.sample(y.shape).to(y)

            loss_kwargs = {}
            if 'std' in batch.keys():
                loss_kwargs['std'] = batch['std'].to(self._device)

            x, y, loss_kwargs = self._pre_model_hook(x, y, batch['dataset'], loss_kwargs)

            prediction = self._model(x)

            loss = self._loss(prediction, y, **loss_kwargs)
            if torch.isnan(loss):
                raise RuntimeError(f'Loss NaN in epoch {epoch}')
            total_loss += loss.item()
            pbar.set_postfix_str(f'Loss: {loss.item():.4f}')
            loss.backward()

            if self._clip_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_gradient_norm)  # type: ignore

            self._tb_logger.log_step(loss=loss.item())

            self._optimizer.step()

            if self._store_path:
                self._training_path.append({name: p.detach().cpu() for name, p in self._model.named_parameters()})

        if not self._cfg.silent:
            LOGGER.info(f'Epoch {epoch} avg loss: {total_loss / len(self._loaders):.4f}')

        if (epoch % self._cfg.eval_every == 0 or epoch % self._cfg.save_every == 0 or epoch == self._n_epochs):
            # store path to be sure we don't lose it if the run gets killed
            if self._store_path:
                for i, step in enumerate(self._training_path):
                    # pretraining model is step -1
                    torch.save(step, str(self._cfg.run_dir /  # type: ignore
                                         f'model_epoch{str(epoch).zfill(3)}_step{i if epoch != 1 else i-1}.p'))
                self._training_path = []

    def _train_sklearn(self):
        assert len(self._loaders) == 1  # make sure we use all data
        data = next(iter(self._loaders))

        x = data['x']
        y = data['y']

        if self.noise_sampler_y is not None:
            y = y + self.noise_sampler_y.sample(y.shape).to(y)

        predict_last_n = self._cfg.predict_last_n
        if predict_last_n is not None:
            x = x[:, -predict_last_n:]
            y = y[:, -predict_last_n:]

        LOGGER.info('Fitting model')
        self._model.fit(x, y)  # type: ignore
        LOGGER.info('Fitting complete')

    def _pre_model_hook(self, x: torch.Tensor, y: torch.Tensor, batch_datasets: List[str], loss_kwargs: dict) \
            -> Tuple[torch.Tensor, torch.Tensor, dict]:
        return x, y, loss_kwargs
