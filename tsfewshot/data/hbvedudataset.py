import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import xarray as xr

from tsfewshot.config import Config
from tsfewshot.data.basedataset import BaseDataset

LOGGER = logging.getLogger(__name__)


class HBVEduDataset(BaseDataset):
    """Dataset for hydrologic HBVedu simulation datasets.

    For this dataset, dataset names must follow the pattern ``fileName#startDate#endDate``, where start and end date
    are accepted by pandas as date representations.

    Parameters
    ---------
    cfg : Config
        Run configuration
    split : {'train', 'val', 'test'}
        Period for which the dataset will be used.
    dataset : str, optional
        If provided, the dataset will ignore the settings in `cfg` and use this dataset instead.
    train_scaler : Dict[str, Dict[str, torch.Tensor]], optional
        Pre-calculated scaler to use for normalization of input/output values.
    train_val_split : float, optional
        float between 0 and 1 to subset the created dataset. If provided, the created dataset will hold a dictionary
        mapping each dataset name to the indices used in the train split. Subsequently, these indices can be used
        to subset the corresponding validation datasets.
    silent : bool, optional
        Option to override cfg.silent.
    """

    def __init__(self, cfg: Config,
                 split: str,
                 dataset: str = None,
                 is_train: bool = True,
                 train_scaler: Dict[str, Dict[str, torch.Tensor]] = None,
                 train_val_split: float = None,
                 silent: bool = False):
        self.stds: Dict[str, torch.Tensor] = {}

        # only allow drawing support sets from the first year of data and query sets from the remaining years.
        self.support_ranges = {}

        super().__init__(cfg, split, dataset=dataset, is_train=is_train, train_scaler=train_scaler,
                         train_val_split=train_val_split, silent=silent)

    def _load_dataset(self, cfg: Config, dataset: str) -> Optional[xr.Dataset]:
        name_components = dataset.split('#')

        # default: first year is reserved for support samples. Can be turned off by adding "#nosupport"
        self.support_ranges[dataset] = 365
        if len(name_components) == 4 and name_components[-1] == 'nosupport':
            self.support_ranges[dataset] = 0
            name_components = name_components[:-1]

        if self._cfg.support_size > self.support_ranges[dataset]:
            raise ValueError('Cannot have support size larger than HBVEduDataset support range')

        if len(name_components) != 3:
            raise ValueError('HBVEduDataset dataset ids must follow the pattern '
                             '"fileName#startDate#endDate[#nosupport]".')
        file_name, start_date, end_date = name_components

        train_file = cfg.base_dir / file_name
        if not train_file.exists():
            LOGGER.warning(f'Dataset {file_name} has no csv file. skipping.')
            return None

        df = pd.read_csv(train_file, index_col=0, parse_dates=[0])

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # start date should be first sample for which we can do predictions, so we need
        # the previous seq_len steps of inputs.
        forcings_start_date = start_date - pd.to_timedelta(self._cfg.seq_length, 'D')
        df = df.loc[forcings_start_date:end_date]
        if df.index[0] > forcings_start_date:
            LOGGER.warning(f'Dataset {dataset} does not have sufficient warmup: first date is {df.index[0]}')
        check_vars = self._input_vars + list(self._target_vars.keys())
        if df[check_vars].isna().any(axis=None):  # type: ignore
            LOGGER.warning(f'Dataset {dataset} has NaNs.')

        # mask out too early target timesteps
        for var in df.columns:
            if var in self._target_vars.keys():
                df.loc[forcings_start_date:start_date - pd.to_timedelta(1, 'D'), var] = np.nan

        df = df.reset_index()

        # for simplicity, only support NSE calculation for the first target
        self.stds[dataset] = torch.tensor(df[list(self._target_vars.keys())[0]].std(skipna=True))

        # BaseDataset expects two dimensions (sample and step), so we consider a basin as one sample with many steps.
        df.index = pd.MultiIndex.from_product([[0], df.index], names=['sample', 'step'])
        xarray = xr.Dataset.from_dataframe(df)
        return xarray

    def __getitem__(self, i: int):
        sample = super().__getitem__(i)
        sample['std'] = self.stds[sample['dataset']]
        return sample
