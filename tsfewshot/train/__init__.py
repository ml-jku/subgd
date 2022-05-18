from tsfewshot.config import Config
from tsfewshot.train.basetrainer import BaseTrainer
from tsfewshot.train.mamltrainer import MAMLTrainer
from tsfewshot.train.metacurvaturetrainer import MetaCurvatureTrainer
from tsfewshot.train.metasgdtrainer import MetaSGDTrainer
from tsfewshot.train.reptiletrainer import ReptileTrainer
from tsfewshot.train.supervisedtrainer import SupervisedTrainer


def get_trainer(cfg: Config, is_finetune: bool = False) -> BaseTrainer:
    """Get a trainer that fits the run configuration.

    Parameters
    ----------
    cfg : Config
        Run configuration
    is_finetune : bool, optional
        Finetune an existing model on a full dataset. The difference to eval with finetune: true is that
        here we don't finetune in a few-shot setting but in normal supervised training with a train and validation set.
    """
    if cfg.training_setup == 'supervised':
        return SupervisedTrainer(cfg, is_finetune=is_finetune)

    if cfg.training_setup == 'maml':
        if cfg.model in ['persistence', 'linear']:
            raise ValueError(f'Model {cfg.model} does not support the maml training setup.')
        return MAMLTrainer(cfg, is_finetune=is_finetune)

    if cfg.training_setup == 'metasgd':
        if cfg.model in ['persistence', 'linear']:
            raise ValueError(f'Model {cfg.model} does not support the metasgd training setup.')
        return MetaSGDTrainer(cfg, is_finetune=is_finetune)

    if cfg.training_setup == 'metacurvature':
        if cfg.model in ['persistence', 'linear']:
            raise ValueError(f'Model {cfg.model} does not support the metacurvature training setup.')
        return MetaCurvatureTrainer(cfg, is_finetune=is_finetune)

    if cfg.training_setup == 'reptile':
        if cfg.model in ['persistence', 'linear']:
            raise ValueError(f'Model {cfg.model} does not support the reptile training setup.')
        return ReptileTrainer(cfg, is_finetune=is_finetune)

    raise ValueError(f'Unknown training setup {cfg.training_setup}.')
