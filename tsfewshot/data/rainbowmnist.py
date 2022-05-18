import logging
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import PIL
import torch
import torchvision
from joblib import Parallel, delayed
from torch.utils import data
from torchvision.transforms import Compose, RandomRotation, Resize, ToTensor
from torchvision.transforms.transforms import CenterCrop, Pad

sys.path.append(str(Path(__file__).parent.parent.parent))
from tsfewshot.config import Config
from tsfewshot.data.imagebasedataset import ImageBaseDataset

LOGGER = logging.getLogger(__name__)

MNIST_IMAGE_SIZE = 28


class RainbowMNISTDataset(ImageBaseDataset):
    """Rainbow MNIST Dataset from [#]_ and [#]_.

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

    References
    ----------
    .. [#] Yao, Huaxiu, Linjun Zhang, and Chelsea Finn. "Meta-Learning with Fewer Tasks through Task Interpolation."
           arXiv preprint arXiv:2106.02695 (2021).
    .. [#] Finn, Chelsea, et al. "Online meta-learning." International Conference on Machine Learning. PMLR, 2019.
    """
    # cache to store a mapping from digit to MNIST indices. We cache this to only calculate it once per digit and not
    # for every dataset.
    SAMPLE_IS_DIGIT = {}

    def __init__(self, cfg: Config, split: str,
                 dataset: str = None,
                 is_train: bool = True,
                 train_scaler: Dict[str, Dict[str, torch.Tensor]] = None,
                 train_val_split: float = None,
                 silent: bool = False):
        self._cfg = cfg
        self._split = split

        # make sure episode generation will work by checking that for
        # each used color/size/rotation, all digits are included.
        if split == 'train':
            split_datasets = cfg.train_datasets
        elif split == 'val':
            split_datasets = cfg.val_datasets
        else:
            split_datasets = cfg.test_datasets
        for ds_name in split_datasets:
            base_dataset = ds_name.rsplit('#', maxsplit=1)[0]
            if any(f'{base_dataset}#{i}' not in split_datasets for i in range(10)):
                raise ValueError('RainbowMNIST dataset experiments must include all digits')

        if any(ds_name.split('#')[1].lower() != 'none' for ds_name in split_datasets) \
                and any(ds_name.split('#')[1].lower() == 'none' for ds_name in split_datasets):
            # colored images would have 3 channels, black-white images only one channel
            raise ValueError('Cannot use colored and non-colored images at the same time')

        super().__init__(cfg, split, dataset=dataset, is_train=is_train, train_scaler=train_scaler,
                         train_val_split=train_val_split, silent=silent)

        self._train_indices = None

    def _load_image_dataset(self, cfg: Config, dataset: str) -> Optional[data.Dataset]:
        """Must return a torch dataset where `__getitem__` yields (image, target) tuples
        and images are of shape (C, H, W). """
        split, color, size, rotation, digit = dataset.split('#')
        digit = int(digit)

        # share SAMPLE_IS_DIGIT across datasets to avoid re-calculating it for every dataset
        if len(RainbowMNISTDataset.SAMPLE_IS_DIGIT) == 0:
            # load once to get the right subset of indices (that belong to the current digit).
            # Don't need to apply any transforms here.
            # In line with Yao et al., MNIST train and test are concatenated to one dataset:
            # https://openreview.net/forum?id=ajXWF7bVR8d&noteId=O4jHvJ00Txa
            mnist = torchvision.datasets.MNIST(root=cfg.base_dir, transform=None, train=True)
            mnist_test = torchvision.datasets.MNIST(root=cfg.base_dir, transform=None, train=False)
            mnist = data.ConcatDataset([mnist, mnist_test])
            for k in range(10):
                LOGGER.info(f"Determining MNIST samples of digit {k}")
                sample_is_digit = Parallel(n_jobs=max(1, cfg.num_workers))(
                    delayed(lambda sample: sample[1] == k)(sample) for sample in mnist)
                sample_is_digit = np.where(sample_is_digit)[0]
                RainbowMNISTDataset.SAMPLE_IS_DIGIT[k] = sample_is_digit
        sample_is_digit = RainbowMNISTDataset.SAMPLE_IS_DIGIT[digit]

        # optionally, subset the dataset even further. This allows specifying the precise indices to use for each task.
        if cfg.dataset_subsets[split] is not None:
            ds_idxs = cfg.dataset_subsets[split][f'{color}#{size}#{rotation}']  # type: ignore
            if ds_idxs is not None:
                # for convenience, we allow specifying the indices as a range like so: "range(0, 4)" -> [0, 1, 2, 3].
                if isinstance(ds_idxs, str):
                    range_bounds = ds_idxs.replace('range(', '').replace(')', '').replace(' ', '').split(',')
                    if len(range_bounds) != 2 or any(i not in '0123456789' for i in range_bounds[0] + range_bounds[1]):
                        raise ValueError(f'Invalid range {range_bounds}. Correct format: "range(start, stop)".')
                    ds_idxs = list(range(int(range_bounds[0]), int(range_bounds[1])))
                LOGGER.info(f'Reducing MNIST task {dataset} to {len(ds_idxs)} samples')
                sample_is_digit = np.intersect1d(sample_is_digit, ds_idxs)
        else:
            LOGGER.warning(f'Using full MNIST dataset for {dataset}')

        # load MNIST with transformations
        transformed_mnist = torchvision.datasets.MNIST(root=cfg.base_dir,
                                                       transform=self._get_transform(color, size, rotation),
                                                       train=True)
        transformed_mnist_test = torchvision.datasets.MNIST(root=cfg.base_dir,
                                                            transform=self._get_transform(color, size, rotation),
                                                            train=False)
        transformed_mnist = data.ConcatDataset([transformed_mnist, transformed_mnist_test])

        # subset to the indices that correspond to the current digit (and optionally to the specified subset of indices)
        return data.Subset(transformed_mnist, sample_is_digit)  # type: ignore

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
        if n_way != 10:
            raise ValueError('RainbowMNIST should be run with n_way == 10')

        # remove digit, keep split, color, size, rotation
        rainbow_types = list(set(ds.rsplit('#', maxsplit=1)[0] for ds in datasets))
        if n_random_combinations is not None and n_random_combinations < len(rainbow_types):
            indices = torch.randperm(len(rainbow_types))[:n_random_combinations]
            rainbow_types = [rainbow_types[i] for i in indices]
        if n_random_combinations is not None and n_random_combinations > len(rainbow_types):
            indices = torch.randint(low=0, high=len(rainbow_types), size=(n_random_combinations,))
            rainbow_types = [rainbow_types[i] for i in indices]

        for rainbow_type in rainbow_types:
            yield tuple(f'{rainbow_type}#{i}' for i in range(10))

    @staticmethod
    def _get_transform(color: str, size: str, rotation: str) -> Compose:
        transforms = []

        if rotation == '0':
            pass
        else:
            try:
                rotation = float(rotation)  # type: ignore
            except ValueError as exception:
                raise ValueError(f'Invalid rotation {rotation}.') from exception
            transforms.append(RandomRotation((rotation, rotation)))

        if color.lower() != 'none':
            transforms.append(_ColorTransform(color))

        resize, pad = None, None
        if size.lower() == 'full':
            pass
        elif size.lower() == 'half':
            resize = MNIST_IMAGE_SIZE // 2
        else:
            try:
                resize = int(size)
            except ValueError as exception:
                raise ValueError(f'Invalid size specification {size}') from exception

        if resize is not None:
            transforms.append(Resize(size=resize))
            if resize < MNIST_IMAGE_SIZE:
                pad = (MNIST_IMAGE_SIZE - resize) // 2
                if resize + 2 * pad != MNIST_IMAGE_SIZE:
                    raise ValueError(
                        f'Resizing to size {resize} and padding with {pad} leads to a different image size.')
                transforms.append(Pad(pad, fill=color if color.lower() != 'none' else 0))
            elif resize > MNIST_IMAGE_SIZE:
                transforms.append(CenterCrop(MNIST_IMAGE_SIZE))
            else:
                pass
        transforms.append(ToTensor())

        return Compose(transforms)


class _ColorTransform:
    """Convert grayscale image to rgb with background of a certain color.

    Parameters
    ----------
    color : str
        Name of the color to use for background pixels.
    """

    def __init__(self, color: str):
        self._color = color

    def __call__(self, image):
        return PIL.ImageOps.colorize(image, self._color, 'white')  # type: ignore


if __name__ == '__main__':

    cfg = Config({
        'base_dir': '/publicdata',
        'run_dir': Path('/tmp'),
        'input_vars': [0],
        'target_vars': {'target': [0]},
        'dataset_max_size': 10,
        'cnn_image_size': 28,
        'classification_n_classes': 10,
        'support_size': 0,
        'query_size': 10,
        'train_datasets': [f'{ds}#{i}' for i in range(10)
                           for ds in ['train#none#full#0', 'train#blue#full#90', 'train#green#full#45']]
    })
    ds = RainbowMNISTDataset(cfg, 'train')

    import matplotlib.pyplot as plt
    for i in np.random.permutation(len(ds)):
        plt.imshow(ds[i]['x'][0].transpose(0, 2))
        print(ds[i]['x'].shape)
        print(ds[i]['x'].min(), ds[i]['x'].max())
        plt.show()
        plt.close()
