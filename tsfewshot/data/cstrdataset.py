import logging
from typing import Dict, Optional

import numpy as np
import torch
import xarray as xr
from tsfewshot.config import Config
from tsfewshot.data.basedataset import BaseDataset

LOGGER = logging.getLogger(__name__)


class CSTRDataset(BaseDataset):
    """Dataset for CSTR trajectories from [#]_.

    Parameters
    ----------
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

    References
    ----------
    .. [#] Forgione, Marco, et al. "On the adaptation of recurrent neural networks for system identification."
           arXiv preprint arXiv:2201.08660 (2022).
    """

    def __init__(self, cfg: Config,
                 split: str,
                 dataset: str = None,
                 is_train: bool = True,
                 train_scaler: Dict[str, Dict[str, torch.Tensor]] = None,
                 train_val_split: float = None,
                 silent: bool = False):
        self.stds = {}
        super().__init__(cfg, split, dataset=dataset, is_train=is_train, train_scaler=train_scaler,
                         train_val_split=train_val_split, silent=silent)

    def _load_dataset(self, cfg: Config, dataset: str) -> Optional[xr.Dataset]:

        ds_file = dataset
        start_slice, end_slice = None, None
        if '#' in dataset:
            ds_spec = dataset.split('#')
            if len(ds_spec) != 3:
                raise ValueError('Must specify dataset as ds#start#end')

            ds_file = ds_spec[0]
            if ds_spec[1] != '':
                start_slice = int(ds_spec[1])
            if ds_spec[2] != '':
                end_slice = int(ds_spec[2])

        train_file = cfg.base_dir / f'{ds_file}'
        if not train_file.exists():
            LOGGER.warning(f'Dataset {dataset} has no npy file. skipping.')
            return None

        all_data = np.load(train_file, allow_pickle=True)

        if cfg.dataset_max_trajectories > 0 and all_data.shape[0] > cfg.dataset_max_trajectories:
            LOGGER.info(f'Limiting dataset {dataset} from {all_data.shape[0]} '
                        f'to {cfg.dataset_max_trajectories} trajectories.')
            all_data = all_data[:cfg.dataset_max_trajectories]

        if start_slice is not None and end_slice is not None:
            all_data = all_data[slice(start_slice, end_slice)]

        if np.isnan(all_data).any():
            LOGGER.warning(f'Dataset {dataset} has NaNs. Skipping.')
            return None

        xarray = xr.Dataset({'q': (['sample', 'step'], all_data[:, :, 0]),
                             't': (['sample', 'step'], all_data[:, :, 1]),
                             'ca': (['sample', 'step'], all_data[:, :, 2]),
                             'cr': (['sample', 'step'], all_data[:, :, 3])
                             })

        return xarray
