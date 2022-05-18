
import pathlib
import logging
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils import data
from torch.utils.data import Dataset
from typing import List, Optional, Dict

from PIL import Image

from tsfewshot.data.imagebasedataset import ImageBaseDataset
from tsfewshot.config import Config

LOGGER = logging.getLogger(__name__)


class MiniImagenetDataset(ImageBaseDataset):
    """ISIC Dataset from [#]_ and [#]_.

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
    silent : bool, optional
        Option to override cfg.silent.

    References
    ----------
    vinyals

    """

    def __init__(self, cfg: Config, split: str,
                 dataset: str = None,
                 is_train: bool = True,
                 train_scaler: Dict[str, Dict[str, torch.Tensor]] = None,
                 train_val_split: float = None,
                 silent: bool = False):

        super().__init__(cfg, split, dataset=dataset, is_train=is_train,
                         train_scaler=train_scaler, train_val_split=train_val_split, silent=silent)

    def _load_image_dataset(self, cfg: Config, dataset: str) -> Optional[data.Dataset]:
        """Must return a torch dataset where `__getitem__` yields (image, target) tuples
        and images are of shape (C, H, W). """
        classes = dataset.split('#')
        split = self._get_split(cfg.base_dir, classes)
        if self._split == 'train':
            all_classes = cfg.train_datasets
        elif self._split == 'val':
            all_classes = cfg.val_datasets
        else:
            all_classes = cfg.test_datasets
        if not cfg.is_label_shared:
            all_classes = classes
        return _MiniImagenetDataset(cfg.base_dir, split, classes, all_classes)

    def _get_split(self, dataset_path: str, classes: List[str]):
        """ Returns the split, in which all classes are. 
        Options are: 'train', 'val', 'test'
        Throws exception, if classes are from different splits."""
        path = pathlib.Path(dataset_path)
        splits = ['train', 'val', 'test']
        split_classes = {}
        for s in splits:
            split_classes[s] = [x.stem for x in (path / s).iterdir()]

        tentative_split = None
        for s, cs in split_classes.items():
            if classes[0] in cs:
                tentative_split = s
        assert tentative_split is not None

        for c in classes:
            if c not in split_classes[tentative_split]:
                raise ValueError('Given classes are not from the same split!')

        return tentative_split


class _MiniImagenetDataset(Dataset):

    def __init__(self, dataset_path: str, dataset_type: str, classes: List[str], all_classes: List[str]):
        """
        Parameters
        ----------
        dataset_path:
            Path to dataset
            dataset folder must be in format:
                /test/(class folders)
                /train/(class folders)
                /val/(class folders)
                /test.csv
                /train.csv
                /val.csv
            where the split is given in the .csv files
        dataset_type:
            must be = 'test', 'train', 'val'
        """
        self.classes = classes
        self.classes2idx = {c: all_classes.index(c) for c in classes}

        self.dataset_path = pathlib.Path(dataset_path)
        self.classes_folder_path = self.dataset_path / dataset_type
        img_label_table_path = self.dataset_path / (dataset_type + '.csv')
        img_table = pd.read_csv(img_label_table_path)
        # read classes list
        self.img_labels = img_table[img_table['label'].isin(classes)].reset_index()
        assert len(self.img_labels.index) != 0

        self.transform = transforms.Compose([
            transforms.Resize([84, 84]),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_labels.index)

    def __getitem__(self, idx):
        classfolder = self.img_labels.loc[idx]['label']
        img_name = self.img_labels.loc[idx]['filename']
        img_path = self.classes_folder_path / classfolder / img_name
        img = Image.open(str(img_path))
        if self.transform:
            img = self.transform(img)
        return img, self.classes2idx[classfolder]
