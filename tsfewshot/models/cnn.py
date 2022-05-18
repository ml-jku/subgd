import torch
from torch import nn

from tsfewshot import utils
from tsfewshot.config import Config
from tsfewshot.models.basemodel import BasePytorchModel
from tsfewshot.models.inputoutputnet import InputOutputNet


class CNN(BasePytorchModel):
    """1D-CNN with input and head layer and dropout between CNN and head.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        if len(cfg.cnn_config) < 1:
            raise ValueError('Need to provide at least one CNN layer configuration')

        self.input_layer = InputOutputNet(cfg, usage='input')

        cnn_layers = []
        for i, layer in enumerate(cfg.cnn_config):
            if i == 0:
                in_channels = self.input_layer.output_size
            else:
                in_channels = cnn_layers[-2].out_channels

            # pad only to the left, so that the kernel doesn't look into the future
            cnn_layers += [nn.ConstantPad1d((layer['kernel_size'] - 1, 0), value=0),
                           nn.Conv1d(in_channels=in_channels,
                                     out_channels=layer['out_channels'], kernel_size=layer['kernel_size']),
                           utils.get_activation(layer['activation'])]

        self.cnn = nn.Sequential(*cnn_layers)

        self.head = InputOutputNet(cfg, usage='output', input_size=cnn_layers[-2].out_channels)
        self._dropout = nn.Dropout(p=cfg.dropout)

    def forward(self, x: torch.Tensor, support_x: torch.Tensor = None, support_y: torch.Tensor = None) -> torch.Tensor:
        x = self.input_layer(x).transpose(1, 2)  # transpose to shape [batch_size, in_channels, seq_length]

        cnn_out = self.cnn(x).transpose(1, 2)  # transpose back to shape [batch_size, seq_length, out_channels]
        return self.head(self._dropout(cnn_out))

    def reset_parameters(self):
        super().reset_parameters()
        self.input_layer.reset_parameters()
        self.head.reset_parameters()
        for i in range(len(self.cnn)):
            module = self.cnn[i]
            if isinstance(module, nn.Conv1d):
                module.reset_parameters()
