from typing import Tuple

import numpy as np
import torch
from torch import nn

from tsfewshot.config import Config
from tsfewshot.models.fc import FC


class InputOutputNet(nn.Module):
    """Optional fully-connected input or output layer network.

    If cfg.input_output_n_random_layers > 1, will randomly use one of n layers in each forward pass.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    usage : {'input', 'output'}
        Whether the net is used as input or output layer
    input_size : int, optional
        Indicates size of the input to the network.
        Required if usage == 'output' or network_spec is provided, ignored otherwise.
    network_spec : dict, optional
        Option to ignore the config settings and create a custom input/output layer. Overrides config specifications.
    """

    def __init__(self, cfg: Config, usage: str, input_size: int = None, network_spec: dict = None):
        super().__init__()

        if network_spec is None:
            if usage.lower() == 'input':
                network_spec = cfg.input_layer
                input_size = len(cfg.input_vars['train'])
                if cfg.input_output_types['input'].lower() == 'both':
                    input_size *= 2  # input deltas and actual values
                elif cfg.cnn_image_size is not None and cfg.model == 'feedforward':
                    input_size = input_size * cfg.cnn_image_size ** 2
                if cfg.encoding_type == 'cat':
                    input_size += cfg.encoding_dim
            elif usage.lower() == 'output':
                network_spec = cfg.output_layer
            else:
                raise ValueError(f'{usage} must be input or output')
        if input_size is None:
            raise ValueError('input_size is required for usage "output" or if network_spec is passed.')
        self.input_size = input_size

        # during inference, use_n_random_nets will be set to 1 to make sure only one net is used for testing.
        self.use_n_random_nets = cfg.input_output_n_random_nets
        self.networks = nn.ModuleList()
        for _ in range(self.use_n_random_nets):
            net, self.output_size = self._get_network(network_spec, input_size)
            self.networks.append(net)

        if usage.lower() == 'output' and self.output_size != cfg.output_size:
            raise ValueError('Output net output size does not match number of target variables')

    @staticmethod
    def _get_network(network_spec: dict, input_size: int) -> Tuple[nn.Module, int]:
        """Get an input or output net following the passed specifications.

        If `network_spec` is None or evaluates to False (empty dict), the returned network is the identity function.

        Parameters
        ----------
        network_spec : dict
            Specification of the network with keys 'hiddens', 'dropout', 'activation', and optionally 'batch_norm'.
        input_size : int
            Size of the inputs into the network.

        Returns
        -------
        Tuple[nn.Module, int]
            The network and its output size.
        """
        if not network_spec:
            return nn.Identity(), input_size

        if input_size < 1:
            raise ValueError('Cannot create network with input size 0')

        hiddens = network_spec['hiddens']
        if len(hiddens) == 0:
            raise ValueError('"hiddens" must be a list of hidden sizes with at least one entry')

        dropout = network_spec['dropout']
        activation = network_spec['activation']

        net = FC(input_size=input_size,
                 hidden_sizes=hiddens,
                 activation=activation,
                 dropout=dropout,
                 batch_norm=network_spec.get('batch_norm', False),
                 output_activation=network_spec.get('output_activation'),
                 flatten=network_spec.get('flatten', False),
                 output_dropout=network_spec.get('output_dropout', False))
        return net, net.output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_n_random_nets > 1:
            net = self.networks[np.random.randint(low=0, high=self.use_n_random_nets)]
        else:
            net = self.networks[0]
        return net(x)

    def reset_parameters(self):
        """Reset the parameters of the input/output layer. """
        for net in self.networks:
            if isinstance(net, FC):
                net.reset_parameters()
