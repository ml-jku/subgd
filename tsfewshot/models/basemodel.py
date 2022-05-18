import logging
from abc import ABC, abstractmethod

import torch
from torch import nn

from tsfewshot.config import Config
from tsfewshot.models.inputoutputnet import InputOutputNet

LOGGER = logging.getLogger(__name__)


class BaseModel(ABC, nn.Module):
    """Abstract base model class.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_pytorch : bool
        Indicates whether the model is a PyTorch model
    """

    def __init__(self, cfg: Config, is_pytorch: bool):
        super().__init__()
        self.is_pytorch = is_pytorch

    @abstractmethod
    def forward(self, x: torch.Tensor, support_x: torch.Tensor = None, support_y: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input to the model
        support_x : torch.Tensor
            Support set input values
        support_y : torch.Tensor
            Support set target values

        Returns
        -------
        torch.Tensor
            Prediction of the model.
        """


class BaseSklearnModel(BaseModel):
    """Abstract base model class for sklear-style models. """

    def __init__(self, cfg: Config):
        super().__init__(cfg, is_pytorch=False)

    @abstractmethod
    def fit(self, x: torch.Tensor, y: torch.Tensor):
        """Fit the linear regression model.

        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (batch_size, seq_length, features)
        y : torch.Tensor
            Target data of same shape as `x`
        """


class BasePytorchModel(BaseModel):
    """Abstract base model class for PyTorch models. """

    def __init__(self, cfg: Config):
        super().__init__(cfg, is_pytorch=True)

        self._cfg = cfg

        self.input_layer: InputOutputNet
        self.head: InputOutputNet
        self.device = cfg.device

    @abstractmethod
    def reset_parameters(self):
        """Reset the parameters of the model. """

    def train(self, mode: bool = True, update_batch_norm_only: bool = False) -> BaseModel:
        """Set model in train or evaluation mode, respecting ``cfg.batch_norm_mode``.

        Parameters
        ----------
        mode : bool, optional
            Whether to set the model to train or eval mode.
        update_batch_norm_only : bool, optional
            If true, only eval mode of batch norm layers will be set according to ``cfg.batch_norm_mode``.

        Returns
        -------
        BaseModel
            The model in train/evaluation mode.
        """
        if not update_batch_norm_only:
            super().train(mode)

        if mode:
            if self._cfg.batch_norm_mode == 'metabn':
                found_batch_norm = False
                for module in self.modules():
                    module_buffers = dict(module.named_buffers()).keys()
                    # set batchnorm layer moment to 1, so only the last mean & var will be stored in the buffers
                    if 'running_mean' in module_buffers and 'running_var' in module_buffers:
                        module.momentum = 1
                        found_batch_norm = True
                if not found_batch_norm:
                    LOGGER.warning(f'Batch norm mode is {self._cfg.batch_norm_mode}, but no batch norm layers found.')

        else:
            if self._cfg.batch_norm_mode in ['transductive', 'maml-conventional', 'metabn']:
                found_batch_norm = False
                for module in self.modules():
                    module_buffers = dict(module.named_buffers()).keys()
                    # keep batchnorm layers in train mode, so they use batch statistics rather than running_mean/_var
                    if 'running_mean' in module_buffers and 'running_var' in module_buffers:
                        if self._cfg.batch_norm_mode == 'transductive':
                            module.train()
                        else:
                            module.eval()
                        found_batch_norm = True
                if not found_batch_norm:
                    LOGGER.warning(f'Batch norm mode is {self._cfg.batch_norm_mode}, but no batch norm layers found.')
            elif self._cfg.batch_norm_mode == 'conventional':
                pass
            else:
                raise ValueError('Invalid BatchNorm mode.')

        return self


class MetaSGDWrapper(BasePytorchModel):
    """Model wrapper for MetaSGD training.

    Adds a per-parameter learnable learning rate to a BasePytorchModel that can be trained
    via MetaSGD. Note that the learning rates are set to ``requires_grad=False``, which
    is changed in the trainer class.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    model : BasePytorchModel
        The model to be wrapped.
    """

    def __init__(self, cfg: Config, model: BasePytorchModel):
        super().__init__(cfg=cfg)
        self.model = model
        lr_range = cfg.maml_inner_lr
        if not isinstance(lr_range, (tuple, list)) or lr_range[0] > lr_range[1]:
            raise ValueError('Must provide a range of inner-loop learning rates to sample from.')
        # trainer will set requires_grad of learning rates
        self.learning_rates = nn.ParameterList([nn.Parameter(torch.rand_like(p) * (lr_range[1] - lr_range[0])
                                                             + lr_range[0],
                                                             requires_grad=False)
                                                for p in model.parameters()])

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def reset_parameters(self):
        return super().reset_parameters()


class MetaCurvatureWrapper(BasePytorchModel):
    """Model wrapper for Meta-Curvature training.

    Adds a per-parameter-matrix preconditioning to a BasePytorchModel that can be trained
    via Meta-Curvature. Note that the preconditioning matrices are set to ``requires_grad=False``, which
    is changed in the trainer class.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    model : BasePytorchModel
        The model to be wrapped.
    """

    def __init__(self, cfg: Config, model: BasePytorchModel):
        super().__init__(cfg=cfg)
        self.model = model

        if any(p.ndim > 2 for p in model.parameters()):
            raise ValueError('MetaCurvature is only implemented for 1- or 2-dim weight matrices (i.e., no convolution)')

        # trainer will set requires_grad of these parameters.
        # m_in uses p.shape[1], because pytorch parameter matrices are of shape (out, in).
        self.m_in = nn.ParameterList([nn.Parameter(torch.eye(p.shape[1], dtype=torch.float32) if p.ndim == 2
                                                   else torch.ones_like(p),
                                                   requires_grad=False)
                                      for p in model.parameters()])
        self.m_out = nn.ParameterList([nn.Parameter(torch.eye(p.shape[0], dtype=torch.float32) if p.ndim == 2
                                                    else torch.tensor(float('nan'), dtype=torch.float32),  # unused
                                                    requires_grad=False)
                                       for p in model.parameters()])

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def reset_parameters(self):
        return super().reset_parameters()

    def toggle_grad(self, grad: bool):
        """Toggle requires_grad for preconditioning matrices. """
        for m in self.m_in:
            m.requires_grad = grad
        for m in self.m_out:
            m.requires_grad = grad
