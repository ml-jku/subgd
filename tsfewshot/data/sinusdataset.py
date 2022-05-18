import logging
from typing import Dict, Iterator, List, Tuple

import torch
from torch.utils.data import Dataset

from tsfewshot.config import Config
from tsfewshot.data.basedataset import BaseDataset

LOGGER = logging.getLogger(__name__)


class SinusDataset(Dataset):
    """Dataset for sinusoid trajectories from [#]_.

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
    .. [#] C. Finn, P. Abbeel, and S. Levine, "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks,"
           in Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia,
           6-11 August 2017, 2017, vol. 70, pp. 1126-1135. http://proceedings.mlr.press/v70/finn17a.html.
    """

    def __init__(self, cfg: Config,
                 split: str,
                 dataset: str = None,
                 is_train: bool = True,
                 train_scaler: Dict[str, Dict[str, torch.Tensor]] = None,
                 train_val_split: float = None,
                 silent: bool = False):
        super().__init__()
        self._cfg = cfg

        if train_val_split is not None:
            raise ValueError('train_val_split not supported for infinite-data sinusoid dataset.')
        if cfg.dataset_max_size < 1:
            raise ValueError('Must set dataset_max_size for infinite-data sinusoid dataset.')

        # dump dummy scaler for compatibility (trainers will use BaseDataset to load scalers from disk)
        if train_scaler is None:
            BaseDataset._dump_scaler(
                cfg.run_dir, BaseDataset.DUMMY_SCALER['mean'], BaseDataset.DUMMY_SCALER['std'])  # type: ignore

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

        self.amp_range = {}
        self.phase_range = {}
        self.freq_range = {}
        self.input_range = {}
        for ds in self._datasets:
            ds_config = self._datasets[0].split('#')
            self.amp_range[ds] = (float(ds_config[0]), float(ds_config[1]))
            self.phase_range[ds] = (float(ds_config[2]), float(ds_config[3]))
            self.input_range[ds] = (float(ds_config[4]), float(ds_config[5]))
            if len(ds_config) >= 7:
                self.freq_range[ds] = (float(ds_config[6]), float(ds_config[7]))
            else:
                self.freq_range[ds] = (1.0, 1.0)

            assert self.amp_range[ds][1] >= self.amp_range[ds][0]
            assert self.phase_range[ds][1] >= self.phase_range[ds][0]
            assert self.input_range[ds][1] >= self.input_range[ds][0]
            assert self.freq_range[ds][1] >= self.freq_range[ds][0]

        self.amp, self.phase, self.freq = {}, {}, {}
        self.draw_parameters()

    def draw_parameters(self):
        """Draw new amplitude and phase parameters for the dataset. """
        for ds in self._datasets:
            self.amp[ds] = torch.rand((1,))[0] * (self.amp_range[ds][1] - self.amp_range[ds][0]) + self.amp_range[ds][0]
            self.phase[ds] = torch.rand((1,))[0] * (self.phase_range[ds][1] -
                                                    self.phase_range[ds][0]) + self.phase_range[ds][0]
            self.freq[ds] = torch.rand((1,))[0] * (self.freq_range[ds][1] -
                                                   self.freq_range[ds][0]) + self.freq_range[ds][0]

    def __len__(self):
        return self._cfg.dataset_max_size * len(self._datasets)

    def __getitem__(self, idx: int):
        ds_idx = torch.randint(low=0, high=len(self._datasets), size=(1,))[0]
        ds = self._datasets[ds_idx]
        ds_name = f'{ds}#{self.amp[ds].item()}#{self.phase[ds].item()}#{self.freq[ds].item()}'

        # shape (1, 1) to have a dummy "time step" dimension in dimension 0
        x = torch.rand((1, 1)) * (self.input_range[ds][1] - self.input_range[ds][0]) + self.input_range[ds][0]

        y = self.amp[ds] * torch.sin(self.freq[ds] * x - self.phase[ds])

        return {'x': x, 'y': y, 'y_base': torch.zeros_like(y), 'dataset': ds_name, 'sample': ds_idx, 'offset': idx}

    def rescale_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Does nothing. Needed for compatibility. """
        return targets

    def is_classification(self) -> bool:
        """Defines whether a dataset is a classification dataset.

        Returns
        -------
        bool
            False
        """
        return False

    def get_valid_dataset_combinations(self, datasets: List[str], n_way: int, n_random_combinations: int = None) \
            -> Iterator[Tuple[str, ...]]:

        if n_way == 1:
            if n_random_combinations is not None:
                if n_random_combinations < len(datasets):
                    random_dataset_indices = torch.randperm(len(datasets))[:n_random_combinations]
                    return iter((datasets[i],) for i in random_dataset_indices)

                random_dataset_indices = torch.randint(low=0, high=len(datasets), size=(n_random_combinations,))
                datasets = [datasets[i] for i in random_dataset_indices]
            return iter((ds,) for ds in datasets)

        raise ValueError('n_way must be 1 for sinusoid dataset')
