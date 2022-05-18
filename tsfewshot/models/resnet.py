from typing import Callable, List

import torch
from torch import nn
from torch.nn import functional as F

from tsfewshot.config import Config
from tsfewshot.models.basemodel import BasePytorchModel
from tsfewshot.models.inputoutputnet import InputOutputNet


class Resnet(BasePytorchModel):
    """Implementation of resnet architecture.

    References
    ----------
    .. [#]  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"
            arXiv:1512.03385 (2015)

    .. [#] https://github.com/akamaster/pytorch_resnet_cifar10

    .. [#] https://d2l.ai/chapter_convolutional-modern/resnet.html 

    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        if cfg.seq_length > 1:
            raise NotImplementedError('Image CNN is only implemented for non-timeseries inputs (i.e., seq_length = 1)')

        #! TODO check for parameterization here!
        # define a parameterization similar to model type imagecnn where each entry in the yaml file corresponds to a Residual layer
        # here: check here if this parameterization is valid
        if not cfg.resnet_config:
            raise ValueError('Need to provide a Resnet configuration')

        # input layer must be identity
        if cfg.input_layer is not None:
            raise NotImplementedError('Input layer not implemented for image inputs.')
        self.input_layer = InputOutputNet(cfg, usage='input')

        output_image_size = cfg.cnn_image_size
        in_channels = self.input_layer.output_size

        #! TODO parameterize here!
        # use the parameterization from above to create the Resnet (so far only two possible parameterizations are hardcoded)
        resnet_type = cfg.resnet_config['resnet_type']
        if resnet_type == 'resnet18-imagenet':
            b1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                               nn.BatchNorm2d(64), nn.ReLU(),
                               nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            option = 'B'
            b2 = _Residual._get_network(64, 64, 2, first_block=True, option=option)
            b3 = _Residual._get_network(64, 128, 2, option=option)
            b4 = _Residual._get_network(128, 256, 2, option=option)
            b5 = _Residual._get_network(256, 512, 2, option=option)
            residuals = [b2, b3, b4, b5]

            self.resnet = nn.Sequential(b1, *residuals, nn.AdaptiveAvgPool2d((1, 1)))

            self.head = InputOutputNet(cfg, usage='output',
                                       input_size=512)
        elif resnet_type == 'resnet20-cifar':
            input_channels_in_resnet = 16
            first_layer = nn.Conv2d(in_channels, input_channels_in_resnet, padding=1, kernel_size=3)

            num_channels = input_channels_in_resnet
            n = 3
            blks = []
            for i in range(3):
                first_block = i == 0
                if first_block:
                    blks.append(_Residual._get_network(num_channels, num_channels,
                                n, first_block=first_block, option='A'))
                else:
                    blks.append(_Residual._get_network(num_channels, 2 *
                                num_channels, n, first_block=first_block, option='A'))
                    num_channels *= 2

            self.resnet = nn.Sequential(first_layer, *blks, nn.AdaptiveAvgPool2d((1, 1)))

            self.head = InputOutputNet(cfg, usage='output',
                                       input_size=64)

        else:
            raise ValueError(f"Unkown resnet_type: {resnet_type}")

        self._dropout = nn.Dropout(p=cfg.dropout)

    def forward(self, x: torch.Tensor, support_x: torch.Tensor = None, support_y: torch.Tensor = None) -> torch.Tensor:
        # time step dim is 1 anyways because we only allow seq_length 1
        x = x[:, 0]  # (batch, channels, h, w)
        x = self.input_layer(x)  # this will just be an identity operation

        resnet_out = self.resnet(x).unsqueeze(1).view(x.shape[0], 1, -1)
        return self.head(self._dropout(resnet_out))

    def reset_parameters(self):
        super().reset_parameters()
        self.input_layer.reset_parameters()
        self.head.reset_parameters()
        for i in range(len(self.resnet)):
            module = self.resnet[i]
            if isinstance(module, _Residual):
                module.reset_parameters()


class _LambdaLayer(nn.Module):
    def __init__(self, lambd: Callable[[torch.Tensor], torch.Tensor]):
        super(_LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambd(x)


class _Residual(nn.Module):
    """The Residual block of ResNet."""

    def __init__(self, input_channels: int, num_channels: int, strides: int = 1, option='A'):
        """
        Parameters
        ----------
        option : str
            either 'A' or 'B'
            refers to the paper Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep Residual Learning for Image Recognition"
            A: pad zeros in skip connections where dimensions do not match
            B: use 1x1 convolutions in skip connections where dimensions do not match
        """
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3,
                               padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                               padding=1)

        self.skip_connect = nn.Sequential()
        if strides != 1 or input_channels != num_channels:
            if option == 'A':
                # num_channels // 4 appends zeros at top and bottom of output channel dimension
                # ::2 takes only every second entry in the feature map
                # i.e. x.shape = [1, 16, 32, 32] gets [1, 32, 16, 16]
                self.skip_connect = _LambdaLayer(lambda x: F.pad(
                    x[:, :, ::2, ::2], (0, 0, 0, 0, num_channels // 4, num_channels // 4), "constant", 0))
            elif option == 'B':
                self.skip_connect = nn.Conv2d(input_channels, num_channels,
                                              kernel_size=1, stride=strides)

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    @staticmethod
    def _get_network(input_channels: int, output_channels: int, num_residuals: int, first_block=False, option='B') -> List[nn.Sequential]:
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(
                    _Residual(input_channels, output_channels, strides=2, option=option))
            else:
                blk.append(_Residual(output_channels, output_channels, option=option))
        return nn.Sequential(*blk)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += self.skip_connect(x)
        return F.relu(y)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        if isinstance(self.skip_connect, nn.Conv2d):
            self.skip_connect.reset_parameters()
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()
