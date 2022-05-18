import logging

import torch
from torch import nn

from tsfewshot.config import Config
from tsfewshot.models.basemodel import BasePytorchModel
from tsfewshot.models.inputoutputnet import InputOutputNet

LOGGER = logging.getLogger(__name__)


class LSTM(BasePytorchModel):
    """LSTM with input and head layer and dropout between LSTM and head.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.input_layer = InputOutputNet(cfg, usage='input')

        self.lstm = nn.LSTM(input_size=self.input_layer.output_size,
                            hidden_size=cfg.hidden_size,
                            num_layers=cfg.lstm_num_layers,
                            batch_first=True)

        self._initial_forget_bias = cfg.lstm_initial_forget_bias
        self._learn_initial_state = cfg.rnn_learn_initial_state
        self.h_0, self.c_0 = None, None
        if self._learn_initial_state:
            if cfg.lstm_num_layers > 1:
                raise ValueError('learn_initial_state is not supported for lstm_num_layers > 1')
            self.h_0 = nn.Parameter(torch.zeros((1, self.lstm.hidden_size)), requires_grad=True)
            self.c_0 = nn.Parameter(torch.zeros((1, self.lstm.hidden_size)), requires_grad=True)

        self.head = InputOutputNet(cfg, usage='output', input_size=cfg.hidden_size)
        self._dropout = nn.Dropout(p=cfg.dropout)

        self.reset_parameters()

    def forward(self, x: torch.Tensor, support_x: torch.Tensor = None, support_y: torch.Tensor = None) -> torch.Tensor:
        x = self.input_layer(x)

        h_x = None
        if self.h_0 is not None:
            batch_size = x.shape[0]
            h_x = self.h_0.repeat(batch_size, 1).unsqueeze(0), self.c_0.repeat(batch_size, 1).unsqueeze(0)
        lstm_out, _ = self.lstm(x, hx=h_x)
        return self.head(self._dropout(lstm_out))

    def reset_parameters(self):
        super().reset_parameters()
        self.input_layer.reset_parameters()
        self.lstm.reset_parameters()
        self.head.reset_parameters()
        if self._learn_initial_state:
            self.h_0.data = torch.zeros((1, self.lstm.hidden_size), device=self.h_0.device)
            self.c_0.data = torch.zeros((1, self.lstm.hidden_size), device=self.c_0.device)
        if self._initial_forget_bias is not None:
            if self.lstm.num_layers > 1:
                LOGGER.warning('initial_forget_bias only affects first LSTM layer')
            self.lstm.bias_hh_l0.data[self.lstm.hidden_size:
                                      2 * self.lstm.hidden_size] = self._initial_forget_bias  # type: ignore
