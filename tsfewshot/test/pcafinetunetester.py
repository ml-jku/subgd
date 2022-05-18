import logging
import pickle
from typing import Optional, Tuple

from torch import optim

from tsfewshot.config import Config
from tsfewshot.logger import Logger
from tsfewshot.models.basemodel import BasePytorchModel
from tsfewshot.test.finetunetester import FinetuneTester
from tsfewshot.utils import get_optimizer_and_scheduler

LOGGER = logging.getLogger(__name__)


class PCAFinetuneTester(FinetuneTester):
    """Class to test a model with PCA finetuning.

    Can only be used after storing PCA values to disk via pca.py.

    Parameters
    ----------
    cfg : Config
        Run configuration
    split : {'train', 'val', 'test'}
        Period to evaluate on.
    init_model_params : bool, optional
        If True, will try to load a model from disk if `model` is None. If False, will use the uninitialized model
        that is returned by ``get_model``.
    tb_logger : Logger, optional
        Initialized tensorboard logger.
    """

    def __init__(self, cfg: Config, split: str, init_model_params: bool = True, tb_logger: Logger = None):
        super().__init__(cfg, split, init_model_params=init_model_params, tb_logger=tb_logger)

        if cfg.finetune_setup != 'pca':
            raise ValueError('Wrong finetune setup for PCAFinetuneTester')

        self._pca = pickle.load(cfg.ig_pca_file.open('rb'))

    def _get_optimizer(self, finetune_model: BasePytorchModel) \
            -> Tuple[optim.Optimizer, Optional[optim.lr_scheduler.MultiStepLR]]:
        return get_optimizer_and_scheduler(self._cfg.optimizer['finetune'],
                                           finetune_model,
                                           self._cfg.finetune_lr[0],
                                           self._cfg.finetune_lr[1],
                                           self._cfg.finetune_lr[2],
                                           weight_decay=self._cfg.weight_decay,
                                           cfg=self._cfg,
                                           pca=self._pca)
