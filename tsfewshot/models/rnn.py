import torch
from torch import nn

from tsfewshot.config import Config
from tsfewshot.models.basemodel import BasePytorchModel
from tsfewshot.models.inputoutputnet import InputOutputNet


class RNN(BasePytorchModel):
    """RNN with input and head layer and dropout between RNN and head.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.input_layer = InputOutputNet(cfg, usage='input')

        self.rnn = nn.RNN(input_size=self.input_layer.output_size,
                          hidden_size=cfg.hidden_size,
                          batch_first=True)

        self._learn_initial_state = cfg.rnn_learn_initial_state
        self.h_0 = None
        if self._learn_initial_state:
            self.h_0 = nn.Parameter(torch.zeros((1, self.rnn.hidden_size)), requires_grad=True)

        self.head = InputOutputNet(cfg, usage='output', input_size=cfg.hidden_size)
        self._dropout = nn.Dropout(p=cfg.dropout)

    def forward(self, x: torch.Tensor, support_x: torch.Tensor = None, support_y: torch.Tensor = None) -> torch.Tensor:
        x = self.input_layer(x)

        h_0 = None
        if self.h_0 is not None:
            batch_size = x.shape[0]
            h_0 = self.h_0.repeat(batch_size, 1).unsqueeze(0)
        rnn_out, _ = self.rnn(x, hx=h_0)
        return self.head(self._dropout(rnn_out))

    def reset_parameters(self):
        super().reset_parameters()
        self.input_layer.reset_parameters()
        self.rnn.reset_parameters()
        self.head.reset_parameters()
        if self._learn_initial_state:
            self.h_0.data = torch.zeros((1, self.rnn.hidden_size), device=self.h_0.device)
