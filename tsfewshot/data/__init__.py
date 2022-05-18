from typing import Dict

import torch

from tsfewshot.config import Config
from tsfewshot.data.basedataset import BaseDataset
from tsfewshot.data.camelsdataset import CamelsDataset
from tsfewshot.data.cstrdataset import CSTRDataset
from tsfewshot.data.hbvedudataset import HBVEduDataset
from tsfewshot.data.odedataset import DoublePendulum, ODEDatasetSimple, ThreeBody
from tsfewshot.data.rainbowmnist import RainbowMNISTDataset
from tsfewshot.data.rlcdataset import RLCDataset
from tsfewshot.data.sinusdataset import SinusDataset
from tsfewshot.data.miniimagenetdataset import MiniImagenetDataset


NAME_TO_DATASET = {'simple_ode': ODEDatasetSimple,
                   'double_pendulum': DoublePendulum, 'three_body': ThreeBody,
                   'hbvedu': HBVEduDataset, 'camels': CamelsDataset,
                   'rainbowmnist': RainbowMNISTDataset,
                   'miniimagenet': MiniImagenetDataset,
                   'cstr': CSTRDataset,
                   'rlc': RLCDataset,
                   'sinusoid': SinusDataset}


def get_dataset(cfg: Config, split: str, dataset: str = None, train_scaler: Dict[str, Dict[str, torch.Tensor]] = None,
                is_train: bool = True, train_val_split: float = None, silent: bool = False) -> BaseDataset:
    """Get dataset according to run configuration.

    Parameters
    ----------
    cfg : Config
        Run configuration.
    split : {'train', 'val', 'test'}
        Period for which the dataset will be used.
    dataset : str, optional
        If provided, the dataset will ignore the settings in `cfg` and use this dataset instead.
    train_scaler : Dict[str, Dict[str, torch.Tensor]], optional
        Pre-calculated scaler to use for normalization of input/output values.
    is_train : bool, optional
        Indicates whether the dataset will be used for training or evaluation (including finetuning).
    train_val_split : float, optional
        float between 0 and 1 to subset the created dataset. If provided, the created dataset will hold a dictionary
        mapping each dataset name to the indices used in the train split. Subsequently, these indices can be used
        to subset the corresponding validation datasets.
    silent : bool, optional
        If True, will not use tqdm progress bars during initialization.

    Returns
    -------
    BaseDataset
        Dataset according to run configuration.

    Raises
    ------
    ValueError
        If `is_train` is False and no `train_scaler` is passed.
    """
    if not is_train and train_scaler is None:
        raise ValueError('Must pass a scaler for evaluation datasets.')

    if cfg.meta_dataset not in NAME_TO_DATASET.keys():
        raise ValueError(f'Unknown meta-dataset {cfg.meta_dataset}.')

    return NAME_TO_DATASET[cfg.meta_dataset](cfg, split=split, dataset=dataset, is_train=is_train,
                                             train_scaler=train_scaler, train_val_split=train_val_split, silent=silent)
