import logging
import sys

from torch.utils.data import DataLoader
from tqdm import tqdm

from tsfewshot.config import Config
from tsfewshot.data import get_dataset
from tsfewshot.data.basedataset import BaseDataset
from tsfewshot.data.episodicdataset import EpisodicDataset, EpisodicOdeDataset
from tsfewshot.train.basetrainer import BaseTrainer

LOGGER = logging.getLogger(__name__)


class EpisodicTrainer(BaseTrainer):
    """Abstract trainer for MAML-style episodic training.

    Parameters
    ----------
    cfg : Config
        Run configuration
    is_finetune : bool, optional
        Finetune an existing model on a full dataset. The difference to eval with finetune: true is that
        here we don't finetune in a few-shot setting but in normal supervised training with a train and validation set.
    """

    def __init__(self, cfg: Config, is_finetune: bool = False):
        super().__init__(cfg, is_finetune=is_finetune)

        self._batch_size = cfg.batch_size

        if not self._model.is_pytorch:
            raise ValueError('Episodic training only supports PyTorch models.')

    def _load_data(self):
        self._datasets = {}
        self._loaders = {}
        self._support_loaders = {}

        LOGGER.info('Loading datasets (split=train)')
        if not self._is_finetune and self._cfg.precalculated_scaler is None:
            # this will dump the scaler to disk so we can use one global scaler for all datasets.
            get_dataset(self._cfg, 'train', is_train=True, silent=True)
        scaler = BaseDataset.load_scaler_from_disk(self._cfg)
        train_ds = self._cfg.train_datasets if not self._is_finetune else self._cfg.val_datasets

        datasets = {}
        for ds in tqdm(train_ds, file=sys.stdout, disable=self._cfg.silent):
            # is_train must be False on finetuning so that we get the averaged layer_per_dataset layer.
            # mode 'finetune' trains on val and evaluates on test. The reason is mostly convenience: this way,
            # we can leave cfg.train_dataset untouched, so layer_per_dataset can still use it to figure out the right
            # size of per-dataset layers.
            dataset = get_dataset(self._cfg,
                                  split='train' if not self._is_finetune else 'val',
                                  dataset=ds,
                                  is_train=not self._is_finetune,
                                  train_scaler=scaler,
                                  silent=True)
            if len(dataset) > 0:
                datasets[ds] = dataset
        if self._cfg.meta_dataset == 'rlc':
            self._datasets = EpisodicOdeDataset(self._cfg, datasets, query_size=self._cfg.maml_inner_batch_size)
        else:
            self._datasets = EpisodicDataset(self._cfg, datasets, query_size=self._cfg.maml_inner_batch_size)

        # use persistent workers so the workers are re-used across episodes (i.e., despite creating new iterators).
        # don't do any batching, because we will use each episode individually
        self._loaders = DataLoader(self._datasets, num_workers=self._cfg.num_workers,
                                   batch_size=None,
                                   persistent_workers=self._cfg.num_workers > 0)
