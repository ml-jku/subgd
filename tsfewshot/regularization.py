import torch
from torch import nn

from tsfewshot.config import Config
from tsfewshot.models.basemodel import BasePytorchModel


class Regularization(nn.Module):
    """Regularization wrapper class that adds up all regularization terms, multiplied by their respective weights.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    model : BasePytorchModel
        The original model, usually before finetuning.
    """

    def __init__(self, cfg: Config, model: BasePytorchModel):
        super().__init__()
        self._reg_fns = []
        self._weights = []
        for reg_spec in cfg.finetune_regularization:
            self._weights.append(reg_spec['weight'])
            if reg_spec['method'] == 'slowweight':
                self._reg_fns.append(SlowWeightRegularization(model, reg_spec.get('norm', 'l2')))  # type: ignore
            else:
                raise ValueError(f'Unknown regularization method {reg_spec["method"]}.')

    def forward(self, model: BasePytorchModel) -> torch.Tensor:
        """Calculate the weighted sum of all regularization terms.

        Parameters
        ----------
        model : BasePytorchModel
            The model to regularize.

        Returns
        -------
        torch.Tensor
            The regularization term.
        """
        reg_term = torch.tensor(0.0, device=model.device)  # pylint: disable=not-callable
        for weight, reg_fn in zip(self._weights, self._reg_fns):
            reg_term = reg_term + weight * reg_fn(model)

        return reg_term


class SlowWeightRegularization(nn.Module):
    """Regularization class that discourages weights from moving far from their original values.

    Parameters
    ----------
    model : BasePytorchModel
        The original model, usually before finetuning.
    norm : str
        Norm to use for calculation of weight differences.
    """

    def __init__(self, model: BasePytorchModel, norm: str):
        super().__init__()
        self._original_params = {name: param.detach().clone() for name, param in model.named_parameters()}
        if norm == 'l1':
            self._regularization_term = lambda a, b: torch.sum(torch.abs(a - b))
        elif norm == 'l2':
            # no sqrt to be as similar as possible to EWC
            self._regularization_term = lambda a, b: torch.sum(torch.square(a - b))
        else:
            raise ValueError(f'Unknown norm {norm}.')

    def forward(self, model: BasePytorchModel) -> torch.Tensor:
        """Calculate the norm of differences between original and current model weights.

        Parameters
        ----------
        model : BasePytorchModel
            The model to regularize.

        Returns
        -------
        torch.Tensor
            The regularization term.
        """
        squared_differences = []
        for name, param in model.named_parameters():
            squared_differences.append(self._regularization_term(self._original_params[name], param))

        return torch.sum(torch.stack(squared_differences))
