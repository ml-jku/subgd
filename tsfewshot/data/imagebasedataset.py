import logging
from abc import abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils import data

from tsfewshot.config import Config
from tsfewshot.data.basedataset import BaseDataset

LOGGER = logging.getLogger(__name__)


class ImageBaseDataset(BaseDataset):
    """Base Dataset for image classification tasks.

    Parameters
    ---------
    cfg : Config
        Run configuration
    split : {'train', 'val', 'test'}
        Period for which the dataset will be used.
    dataset : str, optional
        If provided, the dataset will ignore the settings in `cfg` and use this dataset instead.
    is_train : bool, optional
        Indicates whether the dataset will be used for training or evaluation (including finetuning).
    train_scaler : Dict[str, Dict[str, torch.Tensor]], optional
        Pre-calculated scaler to use for normalization of input/output values.
    train_val_split : float, optional
        float between 0 and 1 to subset the created dataset. If provided, the created dataset will hold a dictionary
        mapping each dataset name to the indices used in the train split. Subsequently, these indices can be used
        to subset the corresponding validation datasets.
    silent : bool, optional
        Option to override cfg.silent.
    """

    def __init__(self, cfg: Config, split: str,
                 dataset: str = None,
                 is_train: bool = True,
                 train_scaler: Dict[str, Dict[str, torch.Tensor]] = None,
                 train_val_split: float = None,
                 silent: bool = False):

        super().__init__(cfg, split, dataset=dataset, is_train=is_train, train_scaler=train_scaler,
                         train_val_split=train_val_split, silent=silent)

    def is_classification(self) -> bool:
        """Defines whether a dataset is a classification dataset. Classification datasets must overwrite this method.

        Returns
        -------
        bool
            True if the dataset is a classification task.
        """
        return True

    def _load_dataset(self, cfg: Config, dataset: str) -> Optional[xr.Dataset]:

        image_dataset = self._load_image_dataset(cfg, dataset)
        if image_dataset is None:
            LOGGER.warning(f'No data for dataset {dataset}.')
            return None

        # this would also happen in BaseDataset, but we save a lot of compute if we do it here,
        # since we won't have to load and transform all of the images.
        if cfg.dataset_max_size > 0 and len(image_dataset) > cfg.dataset_max_size:  # type: ignore
            LOGGER.info(f'Limiting dataset to {cfg.dataset_max_size} images.')
            subset_indices = np.random.permutation(len(image_dataset))[:cfg.dataset_max_size]  # type: ignore
            image_dataset = data.Subset(image_dataset, subset_indices)

        images, targets = [], []
        for image_batch, target_batch in data.DataLoader(image_dataset,
                                                         batch_size=1000,
                                                         shuffle=False,
                                                         num_workers=cfg.num_workers):
            images.append(image_batch)
            targets.append(target_batch)
        images = torch.cat(images, dim=0)
        targets = torch.cat(targets, dim=0)

        if np.isnan(images).any() or np.isnan(targets).any():
            LOGGER.warning(f'Dataset {dataset} has NaNs. Skipping.')
            return None
        if targets.dtype != torch.long and self.is_classification():
            raise ValueError('Target dtype in classification must be long.')

        # images have shape (batch, height, width, channels). We add a time step dimension of 1 for compatibility to
        # the remaining codebase, which generally deals with timeseries.
        images = images.unsqueeze(1)
        targets = targets.unsqueeze(1)

        xarray = xr.Dataset({channel: (['sample', 'step', 'height', 'width'], images[:, :, channel, :, :])
                             for channel in range(images.shape[2])})
        xarray['target'] = (('sample', 'step'), targets)

        if images.shape[3] != images.shape[4]:
            raise ValueError('Non-square images are not supported.')
        if images.shape[3] != cfg.cnn_image_size:
            raise ValueError(f'cfg.cnn_image_size of {cfg.cnn_image_size} is not actual image size {images.shape[3]}.')

        return xarray

    @abstractmethod
    def _load_image_dataset(self, cfg: Config, dataset: str) -> Optional[data.Dataset]:
        """Must return a torch dataset where `__getitem__` yields (image, target) tuples
        and images are of shape (C, H, W). """

    def _normalize_data(self, cfg: Config,
                        scaler_mean: Dict[str, torch.Tensor] = None,
                        scaler_std: Dict[str, torch.Tensor] = None,
                        dump_scaler: bool = True) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Normalize x zero-mean/unit variance.

        If the mean and variance have not yet been calculated (``scaler_mean/std is None``), this method will
        calculate the mean and variance over the whole dataset. Next, it will rescale each input and
        output feature to mean 0 and variance 1.

        Note that `self.y` and `self._y_base` will remain un-normalized for classification tasks.

        Parameters
        ----------
        cfg : Config
            The run configuration.
        scaler_mean : Dict[str, torch.Tensor], optional
            Mean-scaler to use for standardization. Will be calculated if None.
        scaler_std : Dict[str, torch.Tensor], optional
            Std-scaler to use for standardization. Will be calculated if None.
        dump_scaler : bool, optional
            If True and no scaler was passed, the newly calculated scaler will be saved to disk.

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            The mean and variance scalers.
        """
        if scaler_mean is None or scaler_std is None:
            x_global = torch.cat(list(self._x.values()), dim=0).numpy()

            # output features are classes, so we don't want to normalize them.
            scaler_mean = {'x': torch.from_numpy(np.nanmean(x_global, axis=(0, 1, 3, 4))),
                           'y': torch.tensor(0)}  # pylint: disable=not-callable
            scaler_std = {'x': torch.from_numpy(np.nanstd(x_global, axis=(0, 1, 3, 4))),
                          'y': torch.tensor(1)}  # pylint: disable=not-callable
            if not self.is_classification():
                y_global = torch.cat(list(self._y.values()), dim=0).numpy()
                scaler_mean['y'] = torch.from_numpy(np.nanmean(y_global, axis=(0, 1)))
                scaler_std['y'] = torch.from_numpy(np.nanstd(y_global, axis=(0, 1)))

            if (scaler_std['y'] == 0).any():
                raise ValueError('Cannot normalize constant targets.')
            if (scaler_std['x'] == 0).any():
                LOGGER.warning(f'x values have std 0: {torch.nonzero(scaler_std["x"] == 0)}')

            if dump_scaler:
                self._dump_scaler(cfg.run_dir, scaler_mean, scaler_std)  # type: ignore

        # unsqueeze to (batch, time, channels, height, width) to allow auto-broadcasting
        mean = scaler_mean['x'].unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4)
        std = scaler_std['x'].unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4)
        for ds in self._datasets:
            self._x[ds] = (self._x[ds] - mean) / std
            if not self.is_classification():
                self._y[ds] = (self._y[ds] - scaler_mean['y']) / scaler_std['y']

        return scaler_mean, scaler_std
