import math

import torch
from torch import nn

from tsfewshot import utils
from tsfewshot.config import Config
from tsfewshot.models.basemodel import BasePytorchModel
from tsfewshot.models.inputoutputnet import InputOutputNet


class ImageCNN(BasePytorchModel):
    """2D-CNN.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        if cfg.seq_length > 1:
            raise NotImplementedError('Image CNN is only implemented for non-timeseries inputs (i.e., seq_length = 1)')

        if len(cfg.cnn_config) < 1:
            raise ValueError('Need to provide at least one CNN layer configuration')

        # input layer must be identity
        if cfg.input_layer is not None:
            raise NotImplementedError('Input layer not implemented for image inputs.')
        self.input_layer = InputOutputNet(cfg, usage='input')

        cnn_layers = []
        output_image_size = cfg.cnn_image_size
        in_channels = self.input_layer.output_size
        for layer in cfg.cnn_config:
            layer_type = layer.get('type', 'conv').lower()
            if layer_type == 'conv':
                padding = layer.get('padding', 0)
                cnn_layers.append(nn.Conv2d(in_channels=in_channels,
                                            out_channels=layer['out_channels'],
                                            stride=layer.get('stride', 1),
                                            kernel_size=layer['kernel_size'],
                                            padding=padding))  # type: ignore

                output_image_size -= math.ceil(layer['kernel_size'] / 2) - 2 * padding
            elif layer_type == 'feedforward':
                cnn_layers.append(nn.Flatten())
                cnn_layers.append(nn.Linear(in_features=in_channels * output_image_size * output_image_size,
                                            out_features=layer['out_channels'] * (layer['out_features']**2)))
                cnn_layers.append(nn.Unflatten(dim=1, unflattened_size=(layer['out_channels'],
                                                                        layer['out_features'],
                                                                        layer['out_features'])))  # type: ignore

                output_image_size = layer['out_features']
            else:
                raise ValueError(f'Unknown layer type {layer_type}.')

            if layer.get('batch_norm', False):
                cnn_layers.append(nn.BatchNorm2d(layer['out_channels']))
            cnn_layers.append(utils.get_activation(layer['activation']))

            if (max_pool := layer.get('max_pool')) is not None:
                cnn_layers.append(nn.MaxPool2d(max_pool))
                output_image_size = output_image_size // max_pool

            in_channels = layer['out_channels']

        self.cnn = nn.Sequential(*cnn_layers)

        self._dropout = nn.Dropout(p=cfg.dropout)
        self.head = InputOutputNet(cfg, usage='output',
                                   input_size=in_channels * output_image_size * output_image_size)

    def forward(self, x: torch.Tensor, support_x: torch.Tensor = None, support_y: torch.Tensor = None) -> torch.Tensor:
        # time step dim is 1 anyways because we only allow seq_length 1
        x = x[:, 0]  # (batch, channels, h, w)
        x = self.input_layer(x)  # this will just be an identity operation

        cnn_out = self.cnn(x).unsqueeze(1).view(x.shape[0], 1, -1)  # add back time dimension, merge channels and h, w
        return self.head(self._dropout(cnn_out))

    def reset_parameters(self):
        super().reset_parameters()
        self.input_layer.reset_parameters()
        self.head.reset_parameters()
        for i in range(len(self.cnn)):
            module = self.cnn[i]
            if isinstance(module, nn.Conv2d):
                module.reset_parameters()
