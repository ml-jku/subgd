import logging

from tsfewshot.config import Config
from tsfewshot.logger import Logger
from tsfewshot.test.finetunetester import FinetuneTester
from tsfewshot.test.tester import Tester
from tsfewshot.test.pcafinetunetester import PCAFinetuneTester

LOGGER = logging.getLogger(__name__)


def get_tester(cfg: Config, split: str, init_model_params: bool = True, tb_logger: Logger = None) -> Tester:
    """Create a tester according to the run configuration.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    split : {'train', 'val', 'test'}
        Defines which split to test on
    init_model_params : bool, optional
        If True, the tester will try to load a model from disk if `model` is None. If False, it will use the
        uninitialized model that is returned by ``get_model``.
    tb_logger : Logger, optional
        Initialized tensorboard logger.

    Returns
    -------
    Tester
        Tester instance
    """
    if cfg.finetune:
        if cfg.finetune_setup in [None, '']:
            return FinetuneTester(cfg, split, init_model_params=init_model_params, tb_logger=tb_logger)
        if cfg.finetune_setup == 'pca':
            return PCAFinetuneTester(cfg, split, init_model_params=init_model_params, tb_logger=tb_logger)
        raise ValueError(f'Unknown finetune setup {cfg.finetune_setup}')
    if cfg.finetune_modules is not None:
        LOGGER.warning('Ignoring finetune_modules config setting!')

    return Tester(cfg, split, init_model_params=init_model_params, tb_logger=tb_logger)
