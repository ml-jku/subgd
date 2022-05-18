import itertools
import logging
import math
import pickle
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from shutil import copyfile
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import xarray as xr
from numba import njit
from torch.utils.data import Dataset
from tqdm import tqdm

from tsfewshot.config import Config
from tsfewshot.data.utils import PositionalEncoding, RotateFeatures

LOGGER = logging.getLogger(__name__)


class BaseDataset(ABC, Dataset):
    """Generic dataset for timeseries datasets.

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
    # scaler that does nothing. Used (e.g.) for evaluation runs that had no training phase.
    DUMMY_SCALER = {'mean': {'x': torch.tensor(0.0), 'y': torch.tensor(0.0)},  # pylint: disable=not-callable
                    'std': {'x': torch.tensor(1.0), 'y': torch.tensor(1.0)}}  # pylint: disable=not-callable

    def __init__(self, cfg: Config, split: str, dataset: str = None, is_train: bool = True,
                 train_scaler: Dict[str, Dict[str, torch.Tensor]] = None,
                 train_val_split: float = None, silent: bool = False):
        super().__init__()

        self._cfg = cfg
        if not cfg.base_dir.is_dir():
            raise ValueError(f'base_dir {cfg.base_dir} does not exist')

        self._split = split
        if dataset is None:
            if split == 'train':
                self._datasets = cfg.train_datasets
            elif split == 'val':
                self._datasets = cfg.val_datasets
            elif split == 'test':
                self._datasets = cfg.test_datasets
            else:
                raise ValueError(f'Invalid split {split}.')
        else:
            self._datasets = [dataset]

        # this will be populated if train_val_split is not None
        self.train_indices = {ds: None for ds in self._datasets}
        self._train_val_split = train_val_split

        self._dataset_subsets = cfg.dataset_subsets

        self._seq_length = cfg.seq_length
        self._is_train = is_train
        mode = 'train' if is_train else 'finetune'
        self._target_vars = cfg.target_vars[mode]
        self._input_vars = cfg.input_vars[mode]

        self._input_type = cfg.input_output_types['input']
        self._output_type = cfg.input_output_types['output']

        self._layer_per_dataset = cfg.layer_per_dataset if (is_train or cfg.layer_per_dataset_eval) else None
        if self._layer_per_dataset in ['mixedrotation', 'output']:
            dataset = self._datasets[0].split('#rotate')[0]
            if any(ds.split('#rotate')[0] != dataset for ds in self._datasets[1:]):
                raise NotImplementedError('Per-dataset input/output layer is only implemented '
                                          'for different rotations of the same dataset.')
        if self._layer_per_dataset is not None:
            # we need to know the train datasets to figure out the right size of input/output
            # vector if layer_per_dataset is active
            self._train_rotations = [ds.split('#rotate', maxsplit=1)[1] for ds in cfg.train_datasets]
            split_rotations = [ds.split('#rotate', maxsplit=1)[1] for ds in self._datasets]
            if cfg.layer_per_dataset_eval and any(r not in self._train_rotations for r in split_rotations):
                raise ValueError('Cannot evaluate on per-dataset weights for unseen rotations.')
            self._rotation_indices = {ds: self._train_rotations.index(ds.split('#rotate', maxsplit=1)[1])
                                      for ds in self._datasets}

        self._y = {}
        self._y_base = {}
        self._x = {}

        self._scaler_mean = None
        self._scaler_std = None
        if train_scaler is not None:
            self._scaler_mean = train_scaler['mean']
            self._scaler_std = train_scaler['std']

        # for every sample index, store the corresponding dataset and the offset within the dataset
        self._dataset_lookup = []

        if not (silent or cfg.silent):
            LOGGER.info(f'Loading datasets (split={split})')

        self._initialize(cfg, (silent or cfg.silent))

    def _initialize(self, cfg: Config, silent: bool):
        for ds in tqdm(self._datasets, file=sys.stdout, disable=silent):

            rotation_augmentation = None
            if '#rotate' in ds:
                dataset, rotation_spec = ds.split('#rotate', maxsplit=1)
                rotation_augmentation = RotateFeatures(cfg.base_dir, rotation_spec)
            else:
                dataset = ds

            xr_data = self._load_dataset(cfg, dataset)
            if xr_data is None:
                continue

            # make sure the dimensions are ordered (1) sample, (2) step, so that .values will have the right shape
            # most datasets will not have any other dimensions, but image datasets will (height and width dimensions),
            # so we need to pass those additional dimensions to transpose, too, if they are present.
            xr_data = xr_data.transpose(*(['sample', 'step']
                                          + [dim for dim in xr_data.dims
                                             if dim not in ['sample', 'step']]))  # type: ignore

            # Optionally add rotation variable to the dataframe
            if rotation_augmentation is not None:
                xr_data = rotation_augmentation.augment(xr_data)

            x = self._get_inputs(xr_data, self._input_vars)

            # calculate target values.
            # remember previous target values in y_base to be able to derive absolute values from deltas.
            y, y_base = self._get_targets(xr_data, self._target_vars)

            # convert inputs/outputs to deltas if required
            x, y = self._values_to_deltas(x, y, y_base, self._target_vars)

            # cut off first step if inputs contain deltas and last max(target offsets) steps.
            x, y, y_base = self._adjust_length(x, y, y_base, self._target_vars)

            x = torch.from_numpy(x.astype(np.float32))  # type: ignore
            if not self.is_classification():
                y = y.astype(np.float32)
            else:
                if self._is_train and y.max() >= cfg.classification_n_classes['train']:
                    raise ValueError('Number of n classification classes does not match dataset labels.')
                elif not self._is_train and y.max() >= cfg.classification_n_classes['finetune']:
                    raise ValueError('Number of n classification classes does not match dataset labels.')

            y = torch.from_numpy(y)  # type: ignore
            # y_base is unused in classification, so no need to bother about its dtype.
            y_base = torch.from_numpy(y_base.astype(np.float32))  # type: ignore

            # optionally add positional encoding
            if cfg.encoding_type is not None:
                pos_encoder = PositionalEncoding(cfg.encoding_dim, cfg.encoding_type,
                                                 cfg.dropout, max_len=2 * x.shape[1])
                x = pos_encoder(x)

            expected_samples = cfg.support_size + cfg.query_size
            if cfg.timeseries_is_sample:
                if x.shape[0] < expected_samples:
                    LOGGER.warning(f'Dataset {ds} has length {x.shape[0]} < {expected_samples}. Skipping.')
                    continue
                self._dataset_lookup += [(ds, i, None) for i in range(x.shape[0])]

                # for negative offsets, we don't need to cut off the timeseries, because the first steps will be ignored
                # by predict_last_n. We just need to check that the offset doesn't exceed predict_last_n.
                min_offset = 0
                for offsets in self._target_vars.values():
                    int_offsets = [o for o in offsets if isinstance(o, int)]
                    min_offset = min(min_offset, min(int_offsets) if len(int_offsets) > 0 else 0)
                if (min_offset < 0) and (cfg.predict_last_n is None or -min_offset > x.shape[1] - cfg.predict_last_n):
                    raise ValueError(f'Negative target offset {min_offset} exceeds predict_last_n.')

                # sanity check whether predict_last_n is too long for one of the given offsets
                if cfg.predict_last_n is not None:
                    y_last_n = y[:, -cfg.predict_last_n:]
                    y_base_last_n = y_base[:, -cfg.predict_last_n - 1:]
                else:
                    y_last_n, y_base_last_n = y, y_base
                if torch.isnan(y_last_n).any() or torch.isnan(y_base_last_n).any():
                    raise ValueError('y values have NaNs within last predict_last_n timesteps. Check target offsets.')
                if torch.isnan(x).any():
                    raise ValueError(f'x values for dataset {ds} have NaNs.')
            elif self._layer_per_dataset not in ['mixedrotation', 'output'] or ds == self._datasets[0]:
                # with per-dataset input layer in mixed-rotation mode, all datasets are mixed within each sample,
                # so we don't have one sample per dataset/initial condition/offset, but only one sample per
                # initial condition/offset.
                # with multi-head output mode, all datasets are predicted at once, so we also have only one sample
                # per initial condition/offset.

                if cfg.predict_last_n > self._seq_length:
                    raise ValueError('predict_last_n cannot be larger than the input sequence length.')

                samples = _get_valid_samples(x.numpy(), y.numpy(), y_base.numpy(), cfg.predict_last_n, self._seq_length)
                if len(samples) < expected_samples:
                    LOGGER.warning(f'Dataset {ds} has length {len(samples)} < {expected_samples}.')
                    if len(samples) < cfg.support_size or len(samples) == 0:
                        LOGGER.warning(f'Dataset {ds} has not enough samples. Skipping.')
                        continue

                if cfg.dataset_max_size > 0 and len(samples) > cfg.dataset_max_size:
                    LOGGER.info(f'Limiting dataset {ds} from {len(samples)} to {cfg.dataset_max_size} samples.')
                    samples = np.random.permutation(samples)[:cfg.dataset_max_size]

                if self._train_val_split is not None:
                    self.train_indices[ds] = self._get_train_val_split(samples,  # type: ignore
                                                                       self._train_val_split,
                                                                       ds)
                    LOGGER.info(f'Restricting {ds} to {len(self.train_indices[ds])} '  # type: ignore
                                'samples for train/val split.')
                if self._dataset_subsets[self._split] is not None \
                        and (subset_indices := self._dataset_subsets[self._split].get(ds)) is not None:  # type: ignore
                    if max(subset_indices) >= len(samples):
                        raise ValueError(f'Max subset index {max(subset_indices)} >= len samples {len(samples)} '
                                         f'for dataset {ds} in split {self._split}.')
                    LOGGER.info(f'Subsetting dataset {ds} from {len(samples)} to {len(subset_indices)} '
                                f'samples in split {self._split}.')
                    samples = np.array(samples)[subset_indices]

                self._dataset_lookup += [(ds, i, j) for idx, (i, j) in enumerate(samples)
                                         if self.train_indices[ds] is None
                                         or idx in self.train_indices[ds]]  # type: ignore
            else:
                # for per-dataset input layer (mixedrotation or output), we only need to store the samples from
                # the first dataset in the lookup list.
                pass

            self._x[ds] = x
            self._y[ds] = y
            self._y_base[ds] = y_base

        if len(self._x) == 0:
            return  # empty dataset
        self._datasets = sorted(self._x.keys())

        # y_base remains un-normalized, since we only need it to calculate absolute values for metrics, and those
        # metrics will be calculated on un-normalized data.
        self._scaler_mean, self._scaler_std = self._normalize_data(cfg, self._scaler_mean, self._scaler_std,
                                                                   dump_scaler=True)

    def __len__(self):
        return len(self._dataset_lookup)

    def __getitem__(self, i: int):
        dataset, sample, offset = self._dataset_lookup[i]
        if offset is None:
            time_slice = slice(None)  # this is the case where timeseries_is_sample = True
        else:
            time_slice = slice(offset, offset + self._seq_length)

        x = self._x[dataset][sample][time_slice]
        # euler ode expect as 'x' (input to the nn) the x0 initial conditions + the input sequence
        if self._cfg.model == 'eulerode':
            # x is now dictionary containing the keys 'x0' and 'u'
            #! careful: in x0 we are only allowed to pass in the observed state variable
            state_space = self._cfg.eulerode_config['state_space']
            x0 = torch.zeros(size=(1, len(state_space)))
            # fill x0 with observable states
            ts_index_x0 = time_slice.start - 1
            if ts_index_x0 >= 0:  # else: x0 = all zeros
                target_vars = self._cfg.target_vars['train']
                for i, state_var_name in enumerate(state_space):
                    if state_var_name in target_vars:
                        # get target var index to access the variable
                        target_var_names = list(target_vars.keys())
                        target_idx = target_var_names.index(state_var_name)
                        x0[0,i] = self._y[dataset][sample][ts_index_x0, target_idx]  # assign observable part to initial condition
            x = {'x0': x0, 'u': x}
        
        # mixedrotation with 1 dataset is equivalent to singlerotation
        if self._layer_per_dataset == 'mixedrotation' and len(self._datasets) > 1:
            # create an input sequence with random rotations at each step.
            # At each step, all input values are zero except those that belong to the current rotation.
            # Note that this only works if all datasets have the same samples and only differ in their rotation!
            random_datasets = [self._datasets[j] for j in torch.randint(0,
                                                                        len(self._datasets),
                                                                        (self._seq_length, ))]
            x = torch.zeros((x.shape[0], x.shape[1] * len(self._train_rotations)))
            for j, rand_ds in enumerate(random_datasets):
                ds_id = self._rotation_indices[rand_ds]
                x_j = self._x[rand_ds][sample][offset + j]
                # put the current timestep at the position of its rotation
                x[j, ds_id * x_j.shape[0]:(ds_id + 1) * x_j.shape[0]] = x_j
        elif self._layer_per_dataset == 'singlerotation' \
                or (self._layer_per_dataset == 'mixedrotation' and len(self._datasets) == 1):
            # Long input vector where all input values are zero except those that belong to the sample's rotation
            x_temp = torch.zeros((x.shape[0], x.shape[1] * len(self._train_rotations)))
            ds_id = self._rotation_indices[dataset]
            x_temp[:, ds_id * x.shape[1]:(ds_id + 1) * x.shape[1]] = x
            x = x_temp
        if self._layer_per_dataset == 'output':
            # Long target vector where all rotations are predicted at once.
            y = torch.cat([self._y[ds][sample][time_slice] for ds in self._datasets], dim=1)
        else:
            y = self._y[dataset][sample][time_slice]

        return {'x': x,
                'y': y,
                'y_base': self._y_base[dataset][sample][time_slice],
                'dataset': dataset,
                'sample': sample,
                'offset': offset}

    @staticmethod
    def _get_inputs(xr_data: xr.Dataset, input_vars: List[str]) -> np.ndarray:
        """Get input data as numpy array of shape (samples, steps, n_inputs). """
        return np.stack([xr_data[input].values for input in input_vars], axis=2)  # type: ignore

    def _get_targets(self, xr_data: xr.Dataset, target_vars: Dict[str, List[Union[int, str]]]) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Calculate absolute target values from xarray data and offsets.

        Parameters
        ----------
        xr_data : xr.Dataset
            Full dataset to extract target variables from.
        target_vars : Dict[str, List[Union[int, str]]]
            Target variables' offsets or time steps as configured in the config file.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            y (Absolute target values) and y_base (absolute target values to be used as the basis for calculating
            the absolute prediction at step t as ``abs_pred(t) = y_base(t) + delta_pred(t)``).
            If output_type is "deltas", y_base will be absolute y values shifted by one entry. If output_type is
            "delta_t", y_base will be the absolute y value at time t (the last input time step).
            Both y and y_base have shape (samples, steps, n_targets)
        """
        y = []
        y_base = []
        for target, offsets in target_vars.items():
            for offset in offsets:
                if isinstance(offset, str):
                    if offset.startswith('step_'):
                        try:
                            target_step = int(offset[5:])
                        except ValueError as err:
                            raise ValueError(
                                f'String targets must match "step_k", where k is an int. Found {offset}.') from err
                        if target_step == 0 and self._input_type != 'values':
                            LOGGER.warning('Target step 0 will be cut off because input contains deltas.')
                        y_offset: np.ndarray = \
                            np.repeat(np.expand_dims(xr_data[target].sel(step=target_step).values, 1),
                                      len(xr_data['step']),
                                      axis=1)
                        if self._output_type == 'deltas':
                            # delta prediction for constant targets makes no sense, so we set y_base to zero
                            # and always predict absolute values.
                            y_base_offset: np.ndarray = np.zeros_like(y_offset)  # type: ignore
                        elif self._output_type == 'delta_t':
                            y_base_offset = xr_data[target].values
                        else:
                            y_base_offset = np.zeros_like(xr_data[target].values)
                    else:
                        raise ValueError(f'String targets must match "step_k", where k is an int. Found {offset}.')
                elif isinstance(offset, int):
                    y_offset: np.ndarray = np.roll(xr_data[target].values, -offset, axis=1)  # type: ignore
                    if self._output_type == 'deltas':
                        y_base_offset: np.ndarray = np.roll(y_offset, 1, axis=1)  # type: ignore
                    elif self._output_type == 'delta_t':
                        y_base_offset = xr_data[target].values
                    else:
                        y_base_offset = np.zeros_like(xr_data[target].values)

                    if offset > 0:
                        y_offset[:, -offset:] = np.nan
                    elif offset < 0:
                        y_offset[:, :-offset] = np.nan
                        y_base_offset[:, :-offset + 1] = np.nan
                else:
                    raise ValueError(f'Unsupported offset type {type(offset)} (value {offset}).')

                y.append(y_offset)
                y_base.append(y_base_offset)
        y = np.stack(y, axis=2)
        y_base = np.stack(y_base, axis=2)
        return y, y_base  # type: ignore

    def _values_to_deltas(self, x: np.ndarray, y: np.ndarray, y_base: np.ndarray,
                          target_vars: Dict[str, List[Union[int, str]]]) -> Tuple[np.ndarray, np.ndarray]:
        """Optionally convert input/output values to deltas or concatenate deltas and values.

        Parameters
        ----------
        x : np.ndarray
            Absolute input values.
        y : np.ndarray
            Absolute target values.
        y_base : np.ndarray
            Absolute target values from the previous time step (if output_type is "deltas") or from the last input
            time step (if output_type is "delta_t").
        target_vars : Dict[str, List[Union[int, str]]]
            Target variables' offsets or time steps as configured in the config file.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Input values as absolute values, deltas, or both concatenated; target values as absolute values or deltas.
            Constant targets ('step_k') are never converted to deltas (but if output_type is "delta_t", they will be
            converted.).
        """
        if self._output_type in ['deltas', 'delta_t']:
            y_delta = y - y_base

            if self._output_type == 'deltas':
                y_delta[:, 0] = np.nan

                # delta prediction for constant targets makes no sense, so we always predict absolute values for those.
                const_target_idxs = self._get_constant_target_indices(target_vars)
                if len(const_target_idxs) > 0:
                    y_delta[:, :, const_target_idxs] = y[:, :, const_target_idxs]
            y = y_delta
        if self._input_type != 'values':
            x_deltas = x - np.roll(x, 1, axis=1)
            if self._input_type == 'deltas':
                x = x_deltas
            else:
                x = np.concatenate([x, x_deltas], axis=2)  # type: ignore

        return x, y

    def _adjust_length(self, x: np.ndarray, y: np.ndarray, y_base: np.ndarray,
                       target_vars: Dict[str, List[Union[int, str]]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cut off time steps for which we don't have input/output values.

        Parameters
        ----------
        x : np.ndarray
            Input values.
        y : np.ndarray
            Target values.
        y_base : np.ndarray
            Absolute target values to be used to calculate absolute values from deltas.
        target_vars : Dict[str, List[Union[int, str]]]
            Lists of target variables' offsets or time steps.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            `x`, `y`, and `y_base` shortened: First step removed if input contains deltas; last k steps removed, where
            k is the largest target offset.

        Raises
        ------
        ValueError
            If maximum target offset is larger than the sequence length.
        """
        # we don't have a delta for the first step, so we remove that step
        if self._input_type != 'values':
            x = x[:, 1:]
            y = y[:, 1:]
            y_base = y_base[:, 1:]

        # with positive offsets, we don't have targets for the last time steps, so we cut them off.
        max_offset = 0
        for offsets in target_vars.values():
            int_offsets = [o for o in offsets if isinstance(o, int)]
            max_offset = max(max_offset, max(int_offsets) if len(int_offsets) > 0 else 0)
        if max_offset > 0:
            if max_offset > x.shape[1]:
                raise ValueError(f'Positive target offset {max_offset} exceeds sequence length.')
            x = x[:, :-max_offset]
            y = y[:, :-max_offset]
            y_base = y_base[:, :-max_offset]
        return x, y, y_base

    def _normalize_data(self, cfg: Config,
                        scaler_mean: Dict[str, torch.Tensor] = None,
                        scaler_std: Dict[str, torch.Tensor] = None,
                        dump_scaler: bool = True) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Normalize x and y to zero-mean/unit variance.

        If the mean and variance have not yet been calculated (``scaler_mean/std is None``), this method will
        calculate the mean and variance over the whole dataset. Next, it will rescale each input and
        output feature to mean 0 and variance 1.

        Note that `self._y_base` will remain un-normalized.

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
            y_global = torch.cat(list(self._y.values()), dim=0).numpy()

            scaler_mean = {'x': torch.from_numpy(np.nanmean(x_global, axis=(0, 1))),
                           'y': torch.from_numpy(np.nanmean(y_global, axis=(0, 1)))}
            scaler_std = {'x': torch.from_numpy(np.nanstd(x_global, axis=(0, 1))),
                          'y': torch.from_numpy(np.nanstd(y_global, axis=(0, 1)))}

            const_target_idxs = self._get_constant_target_indices(self._target_vars)
            scaler_mean['y'][const_target_idxs] = 0.0
            scaler_std['y'][const_target_idxs] = 1.0

            if (scaler_std['y'] == 0).any():
                raise ValueError('Cannot normalize constant targets.')
            if (scaler_std['x'] == 0).any():
                LOGGER.warning(f'x values have std 0: {torch.nonzero(scaler_std["x"] == 0)}')

            if dump_scaler:
                self._dump_scaler(cfg.run_dir, scaler_mean, scaler_std)  # type: ignore

        if not self.is_classification() and self._y[self._datasets[0]].dtype == torch.long:
            LOGGER.warning('Target dtype is long, but this is not a classification dataset.')

        for ds in self._datasets:
            self._x[ds] = (self._x[ds] - scaler_mean['x']) / scaler_std['x']
            if not self.is_classification():
                self._y[ds] = (self._y[ds] - scaler_mean['y']) / scaler_std['y']

        return scaler_mean, scaler_std

    def rescale_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Undo the zero-mean/unit-variance normalization of `values`.

        Parameters
        ----------
        values : torch.Tensor
            The tensor to rescale.

        Returns
        -------
        torch.Tensor
            The rescaled tensor.

        Raises
        ------
        ValueError
            If this method is called while the scaler is not yet initialized.
        """
        if self.is_classification():
            return targets
        if self._scaler_mean is None or self._scaler_std is None:
            raise ValueError('Cannot rescale targets before initializing the scaler.')
        return targets * self._scaler_std['y'].to(targets) + self._scaler_mean['y'].to(targets)

    def _get_train_val_split(self, samples: List, train_val_split: float, ds_name: str) -> np.ndarray:
        return np.random.choice(range(len(samples)),
                                size=int(train_val_split * len(samples)),
                                replace=False)

    def is_classification(self) -> bool:
        """Defines whether a dataset is a classification dataset. Classification datasets must overwrite this method.

        Returns
        -------
        bool
            True if the dataset is a classification task.
        """
        return False

    @staticmethod
    def _dump_scaler(run_dir: Path, scaler_mean: Dict[str, torch.Tensor], scaler_std: Dict[str, torch.Tensor]):
        """Dump the scaler to disk. """
        if scaler_mean is None or scaler_std is None:
            raise ValueError('Cannot dump uninitialized scaler.')
        scaler_file = run_dir / 'scaler.p'
        if scaler_file.exists():
            raise ValueError(f'Scaler file already exists at {scaler_file}')
        with scaler_file.open('wb') as f:
            pickle.dump({'mean': scaler_mean, 'std': scaler_std}, f)

    @staticmethod
    def load_scaler_from_disk(cfg: Config) -> Dict[str, Dict[str, torch.Tensor]]:
        """Load scaler from disk.

        Parameters
        ----------
        cfg : Config
            Run configuration

        Returns
        -------
        Dict[str, Dict[str, torch.Tensor]]
            The scaler dictionary.
        """
        scaler_path = cfg.run_dir / 'scaler.p'  # type: ignore
        if cfg.precalculated_scaler is not None:
            if scaler_path.exists():
                LOGGER.warning('There already exists a scaler in the run directory. '
                               'Preferring the existing one over the precalculated one.')
            else:
                # copy to run dir so we don't rely on the external scaler to stay at its place
                copyfile(cfg.precalculated_scaler / 'scaler.p', scaler_path)  # type: ignore

        if not scaler_path.exists():
            raise ValueError(f'No scaler found at {scaler_path}')
        with scaler_path.open('rb') as f:
            scaler = pickle.load(f)

        return scaler

    @staticmethod
    def _get_constant_target_indices(target_vars: Dict[str, List[Union[int, str]]]) -> List[int]:
        targets = [target for target_list in target_vars.values() for target in target_list]
        return [i for i, target in enumerate(targets) if isinstance(target, str) and target.startswith('step_')]

    @abstractmethod
    def _load_dataset(self, cfg: Config, dataset: str) -> Optional[xr.Dataset]:
        """Load dataset from file. """

    def get_valid_dataset_combinations(self, datasets: List[str], n_way: int, n_random_combinations: int = None) \
            -> Iterator[Tuple[str, ...]]:
        """Create all or a random subset of all possible N-way combinations of this type of dataset.

        For regression and 1-way datasets, this will simply return the passed list of datasets, or a random subset of
        that list.
        For N-way classification tasks, this will return a list of N-way tasks; either all possible combinations
        (note: this can be a lot of combinations), or a random subset of that list.

        Parameters
        ----------
        datasets : List[str]
            List of datasets to choose combinations from.
        n_way : int
            Defines the N-way setting.
        n_random_combinations : int, optional
            If provided, will only return n random combinations.

        Returns
        -------
        Iterator[Tuple[str, ...]]
            List of possible N-way dataset combinations.
        """
        # using torch, not numpy randomness to make sure there are no unforeseen side effects if this is called
        # from a DataLoader.
        if n_way == 1:
            if n_random_combinations is not None:
                if n_random_combinations < len(datasets):
                    random_dataset_indices = torch.randperm(len(datasets))[:n_random_combinations]
                    return iter((datasets[i],) for i in random_dataset_indices)

                random_dataset_indices = torch.randint(low=0, high=len(datasets), size=(n_random_combinations,))
                datasets = [datasets[i] for i in random_dataset_indices]
            return iter((ds,) for ds in datasets)

        if len(datasets) - n_way < 0:
            raise ValueError(f'Cannot build {n_way}-way episodes from {len(datasets)} datasets.')
        n_total_combinations = math.factorial(len(datasets)) / math.factorial(len(datasets) - n_way)
        if self._cfg.sort_episode_classes:
            n_total_combinations /= math.factorial(n_way)
        if n_random_combinations is not None:
            # if n_random_combinations < n_total_combinations, we can make sure that each combination only appears once.
            if n_random_combinations < n_total_combinations:
                random_combinations = set()
                while len(random_combinations) < n_random_combinations:
                    indices = list(torch.randperm(len(datasets))[:n_way])
                    if self._cfg.sort_episode_classes:
                        indices = sorted(indices)
                    random_combinations.add(tuple(datasets[i] for i in indices))
            else:
                random_combinations = []
                for _ in range(n_random_combinations):
                    indices = list(torch.randperm(len(datasets))[:n_way])
                    if self._cfg.sort_episode_classes:
                        indices = sorted(indices)
                    random_combinations.append(tuple(datasets[i] for i in indices))
            return iter(random_combinations)

        if n_total_combinations > 1000:
            LOGGER.warning('There is a large number of possible dataset combinations. Consider restricting them.')
        return itertools.combinations(datasets, n_way)


@njit
def _get_valid_samples(x: np.ndarray, y: np.ndarray, y_base: np.ndarray,
                       predict_last_n: int, seq_length: int) -> List[Tuple[int, int]]:
    samples = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1] - seq_length + 1):
            target_time_slice = slice(j + seq_length - predict_last_n, j + seq_length)
            if np.isfinite(x[i, j:j + seq_length]).all() \
                    and np.isfinite(y[i, target_time_slice]).all() \
                    and np.isfinite(y_base[i, target_time_slice]).all():
                samples.append((i, j))
    return samples
