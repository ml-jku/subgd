"""
Some of this code is copied from the neuralHydrology Python package (https://github.com/neuralhydrology/neuralhydrology)
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
import xarray as xr

from tsfewshot.config import Config
from tsfewshot.data.basedataset import BaseDataset

LOGGER = logging.getLogger(__name__)


class CamelsDataset(BaseDataset):
    """Dataset for hydrologic CAMELS US dataset by [#]_ and [#]_.

    For this dataset, dataset names must follow the pattern ``basinName#startDate#endDate``, where start and end date
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

    References
    ----------
    .. [#] A. J. Newman, M. P. Clark, K. Sampson, A. Wood, L. E. Hay, A. Bock, R. J. Viger, D. Blodgett,
        L. Brekke, J. R. Arnold, T. Hopson, and Q. Duan: Development of a large-sample watershed-scale
        hydrometeorological dataset for the contiguous USA: dataset characteristics and assessment of regional
        variability in hydrologic model performance. Hydrol. Earth Syst. Sci., 19, 209-223, 
        doi:10.5194/hess-19-209-2015, 2015
    .. [#] Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P.: The CAMELS data set: catchment attributes and
        meteorology for large-sample studies, Hydrol. Earth Syst. Sci., 21, 5293-5313, doi:10.5194/hess-21-5293-2017,
        2017.
    """

    def __init__(self, cfg: Config,
                 split: str,
                 dataset: str = None,
                 is_train: bool = True,
                 train_scaler: Dict[str, Dict[str, torch.Tensor]] = None,
                 train_val_split: float = None,
                 silent: bool = False):
        self.stds: Dict[str, torch.Tensor] = {}
        super().__init__(cfg, split, dataset=dataset, is_train=is_train, train_scaler=train_scaler,
                         train_val_split=train_val_split, silent=silent)

    def _load_dataset(self, cfg: Config, dataset: str) -> Optional[xr.Dataset]:
        name_components = dataset.split('#')
        if len(name_components) != 3:
            raise ValueError('CamelsDataset dataset ids must follow the pattern "basinName#startDate#endDate".')
        basin_name, start_date, end_date = name_components

        # get forcings
        df, area = self._load_camels_us_forcings(cfg.base_dir, basin_name, 'daymet')

        # add discharge
        df['QObs(mm/d)'] = self._load_camels_us_discharge(cfg.base_dir, basin_name, area)

        # replace invalid discharge values by NaNs
        qobs_cols = [col for col in df.columns if 'qobs' in col.lower()]
        for col in qobs_cols:
            df.loc[df[col] < 0, col] = np.nan

        df = df.loc[start_date:end_date]

        # add static attributes
        attributes = self._load_camels_us_attributes(cfg.base_dir, [basin_name])
        for attr_name in attributes.columns:
            df[attr_name] = attributes.loc[basin_name, attr_name]

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

    @staticmethod
    def _load_camels_us_attributes(data_dir: Path, basins: List[str]) -> pd.DataFrame:
        """Load CAMELS US attributes from the dataset provided by [#]_.

        Parameters
        ----------
        data_dir : Path
            Path to the CAMELS US directory. This folder must contain a 'camels_attributes_v2.0' folder (the original
            data set) containing the corresponding txt files for each attribute group.
        basins : List[str], optional
            If passed, return only attributes for the basins specified in this list. Otherwise, the attributes of all
            basins are returned.

        Returns
        -------
        pandas.DataFrame
            Basin-indexed DataFrame, containing the attributes as columns.

        References
        ----------
        .. [#] Addor, N., Newman, A. J., Mizukami, N. and Clark, M. P.: The CAMELS data set: catchment attributes and
            meteorology for large-sample studies, Hydrol. Earth Syst. Sci., 21, 5293-5313,
            doi:10.5194/hess-21-5293-2017, 2017.
        """
        attributes_path = data_dir / 'camels_attributes_v2.0'

        if not attributes_path.exists():
            raise RuntimeError(f"Attribute folder not found at {attributes_path}")

        txt_files = attributes_path.glob('camels_*.txt')

        # Read-in attributes into one big dataframe
        dfs = []
        for txt_file in txt_files:
            df_temp = pd.read_csv(txt_file, sep=';', header=0, dtype={'gauge_id': str})
            df_temp = df_temp.set_index('gauge_id')

            dfs.append(df_temp)

        df = pd.concat(dfs, axis=1)
        # convert huc column to double digit strings
        df['huc'] = df['huc_02'].apply(lambda x: str(x).zfill(2))
        df = df.drop('huc_02', axis=1)

        if basins:
            if any(b not in df.index for b in basins):
                raise ValueError('Some basins are missing static attributes.')
            df = df.loc[basins]  # type: ignore

        return df

    @staticmethod
    def _load_camels_us_forcings(data_dir: Path, basin: str, forcings: str) -> Tuple[pd.DataFrame, int]:
        """Load the forcing data for a basin of the CAMELS US data set.

        Parameters
        ----------
        data_dir : Path
            Path to the CAMELS US directory. This folder must contain a 'basin_mean_forcing' folder containing one
            subdirectory for each forcing. The forcing directories have to contain 18 subdirectories (for the 18 HUCS)
            as in the original CAMELS data set. In each HUC folder are the forcing files (.txt), starting with the
            8-digit basin id.
        basin : str
            8-digit USGS identifier of the basin.
        forcings : str
            Can be e.g. 'daymet' or 'nldas', etc. Must match the folder names in the 'basin_mean_forcing' directory. 

        Returns
        -------
        pd.DataFrame
            Time-indexed DataFrame, containing the forcing data.
        int
            Catchment area (m2), specified in the header of the forcing file.
        """
        forcing_path = data_dir / 'basin_mean_forcing' / forcings
        if not forcing_path.is_dir():
            raise OSError(f"{forcing_path} does not exist")

        files = list(forcing_path.glob('**/*_forcing_leap.txt'))
        file_path = [f for f in files if f.name[:8] == basin]
        if file_path:
            file_path = file_path[0]
        else:
            raise FileNotFoundError(f'No file for Basin {basin} at {forcing_path}')

        df = pd.read_csv(file_path, sep='\s+', header=3)
        df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) +
                                    "/" + df.Day.map(str), format="%Y/%m/%d")
        df = df.set_index("date")

        # load area from header
        with open(file_path, 'r') as fp:
            content = fp.readlines()
            area = int(content[2])

        return df, area

    @staticmethod
    def _load_camels_us_discharge(data_dir: Path, basin: str, area: int) -> pd.Series:
        """Load the discharge data for a basin of the CAMELS US data set.

        Parameters
        ----------
        data_dir : Path
            Path to the CAMELS US directory. This folder must contain a 'usgs_streamflow' folder with 18
            subdirectories (for the 18 HUCS) as in the original CAMELS data set. In each HUC folder are the discharge
            files (.txt), starting with the 8-digit basin id.
        basin : str
            8-digit USGS identifier of the basin.
        area : int
            Catchment area (m2), used to normalize the discharge.

        Returns
        -------
        pd.Series
            Time-index pandas.Series of the discharge values (mm/day)
        """

        discharge_path = data_dir / 'usgs_streamflow'
        files = list(discharge_path.glob('**/*_streamflow_qc.txt'))
        file_path = [f for f in files if f.name[:8] == basin]
        if file_path:
            file_path = file_path[0]
        else:
            raise FileNotFoundError(f'No file for Basin {basin} at {file_path}')

        col_names = ['basin', 'Year', 'Mnth', 'Day', 'QObs', 'flag']
        df = pd.read_csv(file_path, sep='\s+', header=None, names=col_names)  # type: ignore
        df["date"] = pd.to_datetime(df.Year.map(str) + "/" + df.Mnth.map(str) +
                                    "/" + df.Day.map(str), format="%Y/%m/%d")
        df = df.set_index("date")

        # normalize discharge from cubic feet per second to mm per day
        df.QObs = 28316846.592 * df.QObs * 86400 / (area * 10**6)

        return df.QObs
