from abc import ABC, abstractmethod

import torch
from torch import nn

from tsfewshot.config import Config


class BaseLoss(ABC, nn.Module):
    """Abstract base loss class.

    Calculates loss on the last ``cfg.predict_last_n`` time steps. (All steps if ``cfg.predict_last_n is None``)

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self._predict_last_n = cfg.predict_last_n

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        assert prediction.shape == target.shape
        if self._predict_last_n is not None:
            prediction, target = prediction[:, -self._predict_last_n:], target[:, -self._predict_last_n:]

        return self._loss_fn(prediction, target, **kwargs)

    @abstractmethod
    def _loss_fn(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        pass


class MSELoss(BaseLoss):
    """Mean Squared Error loss.

    Calculates MSE on the last ``cfg.predict_last_n`` time steps. (All steps if ``cfg.predict_last_n is None``)

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._loss = nn.MSELoss()

    def _loss_fn(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._loss(prediction, target)


class MAELoss(BaseLoss):
    """Mean Absolute Error loss.

    Calculates MAE on the last ``cfg.predict_last_n`` time steps. (All steps if ``cfg.predict_last_n is None``)

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._loss = nn.L1Loss()

    def _loss_fn(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:
        return self._loss(prediction, target)


class NSELoss(BaseLoss):
    """Nash--Sutcliffe Efficiency Loss from [#]_.

    Calculates an adapted version of NSE that can be optimized with Gradient Descent.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    eps : float, optional
        Epsilon for numerical stability.

    References
    ----------
    .. [#] Kratzert, Frederik, et al. "Towards learning universal, regional, and local hydrological behaviors
           via machine learning applied to large-sample datasets."
           Hydrology and Earth System Sciences 23.12 (2019): 5089-5110.
    """

    def __init__(self, cfg: Config, eps: float = 0.1):
        super().__init__(cfg)
        self._eps = eps
        if cfg.meta_dataset not in ['camels', 'hbvedu']:
            raise ValueError('Dataset does not provide std information.')

    def _loss_fn(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs) \
            -> torch.Tensor:
        per_basin_target_stds = kwargs['std']
        # expand dimension 1 to predict_last_n
        per_basin_target_stds = per_basin_target_stds.unsqueeze(1).unsqueeze(1).expand_as(prediction)

        squared_error = (prediction - target)**2
        weights = 1 / (per_basin_target_stds + self._eps)**2
        scaled_loss = weights * squared_error
        return torch.mean(scaled_loss)


class CrossEntropyLoss(BaseLoss):
    """Crossentropy loss.

    Only supported for ``cfg.predict_last_n == 1``.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        if cfg.predict_last_n > 1:
            raise ValueError('Crossentropy loss only supported for single-step prediction.')

        super().__init__(cfg)
        self._loss = nn.CrossEntropyLoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # last dim in prediction is n_classes, last dim in target is 1
        assert prediction.shape[:-1] == target.shape[:-1]
        assert target.shape[2] == 1

        # can directly choose -1 in dim 1, because we only allow predict_last_n == 1.
        # dim 2 in target will always have shape 1, because classes are different integers in that dimension.
        return self._loss_fn(prediction[:, -1], target[:, -1, 0])

    def _loss_fn(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(prediction, target)


def get_loss(cfg: Config) -> BaseLoss:
    """Get loss module.

    Parameters
    ----------
    cfg : Config
        The run configuration.

    Returns
    -------
    BaseLoss
        The loss module.
    """
    if cfg.loss_fn == 'mae':
        return MAELoss(cfg)
    if cfg.loss_fn == 'mse':
        return MSELoss(cfg)
    if cfg.loss_fn == 'nse':
        return NSELoss(cfg)
    if cfg.loss_fn == 'crossentropy':
        return CrossEntropyLoss(cfg)

    raise ValueError(f'Unknown loss function {cfg.loss_fn}.')
