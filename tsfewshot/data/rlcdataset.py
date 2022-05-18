import logging
from typing import Dict, Optional

import numpy as np
import torch
import xarray as xr
from tsfewshot.config import Config
from tsfewshot.data.basedataset import BaseDataset

LOGGER = logging.getLogger(__name__)


class RLCDataset(BaseDataset):
    """Dataset for RLC trajectories from [#]_.

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
            else: 
                start_slice = 0
            if ds_spec[2] != '':
                end_slice = int(ds_spec[2])
            else:
                end_slice = None

        train_file = cfg.base_dir / f'{ds_file}'
        if not train_file.exists():
            LOGGER.warning(f'Dataset {dataset} has no npy file. skipping.')
            return None

        all_data = np.load(train_file, allow_pickle=True)

        if cfg.dataset_max_trajectories > 0 and all_data.shape[0] > cfg.dataset_max_trajectories:
            LOGGER.info(f'Limiting dataset {dataset} from {all_data.shape[0]} '
                        f'to {cfg.dataset_max_trajectories} trajectories.')
            all_data = all_data[:cfg.dataset_max_trajectories]
          
        ds_slice = slice(start_slice, end_slice)
        all_data = all_data[ds_slice]

        if np.isnan(all_data).any():
            LOGGER.warning(f'Dataset {dataset} has NaNs. Skipping.')
            return None

        # TODO make this configurable
        # scale data according to Forgione paper code
        # scale vin (input voltage)
        all_data[:, :, 1] /= 80.
        # scale vc (output voltage)
        all_data[:, :, 2] /= 90.
        # scale il (inductor current)
        all_data[:, :, 3] /= 3.

        # add output noise according to Forgione paper code
        noise_std = 0.1
        all_data[:, :, 2] += np.random.randn(*all_data[:, :, 2].shape) * noise_std

        # Forgione paper does a second scaling here, which has no effect in final performance.
        # For this reason it is skipped, intentionally.
        if ds_slice.start is not None and ds_slice.stop is not None and self._cfg.plot_n_figures > 0:
            self._plot_dataset(cfg, dataset, ds_slice, all_data)

        xarray = xr.Dataset({'te': (['sample', 'step'], all_data[:, :, 0]),
                             'vin': (['sample', 'step'], all_data[:, :, 1]),
                             'vc': (['sample', 'step'], all_data[:, :, 2]),
                             'il': (['sample', 'step'], all_data[:, :, 3])
                             })

        return xarray

    def _plot_dataset(self, cfg: Config, dataset: str, ds_slice: slice, data: np.ndarray):
        import matplotlib.pyplot as plt
        from pathlib import Path
        ds_name = Path(dataset).stem
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
        labels = ['vin', 'vc', 'il']
        for i, ax in enumerate(axes):
            for traj_idx in range(data.shape[0]):
                ax.plot(data[traj_idx, :, i + 1], label=f'sample_{ds_slice.start+traj_idx}')  # i+1 since we skip time dimension
                ax.set_ylabel(f'{labels[i]}')
                ax.grid(True)
            if i == len(axes) - 1:
                ax.legend(frameon=False, loc=(1, 0))
                ax.set_xlabel('step')
        fig.suptitle(f'dataset_{ds_name}_traj_slice({ds_slice.start},{ds_slice.stop})')
        fig.savefig(
            str(cfg.run_dir / f'dataset_{ds_name}#{ds_slice.start}#{ds_slice.stop}.jpg'), dpi=200, bbox_inches="tight")
        plt.close()
