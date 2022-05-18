from typing import List

import torch
from torch import nn

from tsfewshot.utils import get_activation


class FC(nn.Module):
    """Fully-connected network.

    Parameters
    ----------
    input_size : int
        Input size of the network.
    hidden_sizes : List[int]
        Hidden sizes of the network. The last entry will be the output size.
    activation : str
        Activation of the network.
    dropout : float
        Dropout between network layers.
    batch_norm : bool, optional.
        If True, will use BatchNorm between linear layers and activation functions.
    output_activation : str, optional
        If provided, will use this activation function after the last layer.
    flatten : bool, optional
        If True, will add a flatten layer before first linear layers.
    output_dropout : bool, optional
        If True, adds a dropout layer after the output activation (if specified). 
        (Can be useful if there is another module, i.e. another inputoutputnet after this.)
    """

    def __init__(self, input_size: int,
                 hidden_sizes: List[int],
                 activation: str,
                 dropout: float,
                 batch_norm: bool = False,
                 output_activation: str = None,
                 flatten: bool = False,
                 output_dropout: bool = False):
        super().__init__()

        self.input_size = input_size
        self.output_size = hidden_sizes[-1]
        hidden_sizes = hidden_sizes[:-1]

        layers = []
        if len(hidden_sizes) > 0:
            if flatten:
                # input data is of shape BxTx(...), where B is batch dimension and T is time dimension
                # we keep these first two dimensions to avoid error in loss function with misaligned target shape
                layers.append(nn.Flatten(start_dim=2))
            for i, hidden_size in enumerate(hidden_sizes):
                # bias before BatchNorm is useless noop.
                if i == 0:
                    layers.append(nn.Linear(input_size, hidden_size, bias=not batch_norm))
                else:
                    layers.append(nn.Linear(hidden_sizes[i - 1], hidden_size, bias=not batch_norm))

                if batch_norm:
                    layers.append(_TransposeTimeFeatures())
                    layers.append(nn.BatchNorm1d(hidden_size))
                    layers.append(_TransposeTimeFeatures())
                layers.append(get_activation(activation))
                layers.append(nn.Dropout(p=dropout))

            layers.append(nn.Linear(hidden_sizes[-1], self.output_size))
        else:
            layers.append(nn.Linear(input_size, self.output_size))

        if output_activation is not None:
            layers.append(get_activation(output_activation))
        if output_dropout:
            layers.append(nn.Dropout(p=dropout))
        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def reset_parameters(self):
        """Reset the parameters of the FC network. """
        for i in range(len(self.fc)):
            module = self.fc[i]
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class _TransposeTimeFeatures(nn.Module):
    """Helper module that transposes time and feature dimensions.

    Needed for BatchNorm1d layers, which expect (batch, features, time), whereas
    everywhere else we have (batch, time, features).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2)
