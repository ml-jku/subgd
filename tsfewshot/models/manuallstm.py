from __future__ import annotations

import copy
import logging
import math
from typing import Dict, List, Union

import torch
from torch import nn

from tsfewshot.config import Config
from tsfewshot.models.basemodel import BasePytorchModel
from tsfewshot.models.inputoutputnet import InputOutputNet
from tsfewshot.models.lstm import LSTM

LOGGER = logging.getLogger(__name__)


class ManualLSTM(BasePytorchModel):
    """An LSTM implementation that doesn't use nn.LSTM, gives access to all gates and states and allows multiple gates.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    is_finetune : bool, False
        Indicates whether the model is used for training or finetuning.
        This can have an effect on the inputs to the head.
    """

    def __init__(self, cfg: Config, is_finetune: bool = False):
        super().__init__(cfg)

        mode = 'finetune' if is_finetune else 'train'

        self._hidden_size = cfg.hidden_size

        # If true, use one more input (the error from the last time step)
        self._error_input = cfg.lstm_finetune_error_input
        if self._error_input and not cfg.timeseries_is_sample:
            raise ValueError('lstm_finetune_error_input only possible with timeseries_is_sample = True.')

        self.input_layer = InputOutputNet(cfg, usage='input')

        self.lstm = _LSTMCell(input_size=self.input_layer.output_size,
                              hidden_size=self._hidden_size,
                              error_size=int(self._error_input) * cfg.output_size,
                              initial_forget_bias=cfg.lstm_initial_forget_bias,
                              n_i_gates=cfg.lstm_n_gates[mode]['i'],
                              n_f_gates=cfg.lstm_n_gates[mode]['f'],
                              n_g_gates=cfg.lstm_n_gates[mode]['g'],
                              n_o_gates=cfg.lstm_n_gates[mode]['o'])

        self._learn_initial_state = cfg.rnn_learn_initial_state
        self.h_0, self.c_0 = None, None
        if self._learn_initial_state:
            self.h_0 = nn.Parameter(torch.zeros((1, self.lstm.hidden_size)), requires_grad=True)
            self.c_0 = nn.Parameter(torch.zeros((1, self.lstm.hidden_size)), requires_grad=True)

        self._head_inputs = cfg.lstm_head_inputs[mode]
        if self._head_inputs == ['h'] and not self._error_input \
                and self.lstm._n_i_gates == self.lstm._n_f_gates == self.lstm._n_g_gates == self.lstm._n_o_gates == 1:
            LOGGER.info('Only head input is hidden state, no additional gates, no error input. Could use normal LSTM.')
        self.head = InputOutputNet(cfg, usage='output', input_size=len(self._head_inputs) * self._hidden_size)

        self._dropout = nn.Dropout(p=cfg.dropout)

    def forward(self, x: torch.Tensor, support_x: torch.Tensor = None,
                support_y: torch.Tensor = None, y: torch.Tensor = None) -> torch.Tensor:
        """Perform a forward pass through the LSTM.

        Parameters
        ----------
        x : torch.Tensor
            Input timeseries of shape [batch_size, seq_length, input_size]
        support_x : torch.Tensor, optional
            Ignored.
        support_y : torch.Tensor, optional
            Ignored.
        y : torch.Tensor, optional
            Target values to calculate the error on predicted timesteps, necessary if
            cfg.lstm_fewshot_error_input is True: In this case, the LSTM will take its error on the previous timestep
            as an additional input.

        Returns
        -------
        torch.Tensor
            Prediction of shape [batch_size, seq_length, output_size].
        """
        x = self.input_layer(x)

        batch_size, seq_len, _ = x.shape

        if self.h_0 is not None:
            h_0, c_0 = self.h_0.repeat(batch_size, 1), self.c_0.repeat(batch_size, 1)
        else:
            h_0 = torch.zeros((batch_size, self._hidden_size)).to(x)
            c_0 = torch.zeros((batch_size, self._hidden_size)).to(x)
        h_x = (h_0, c_0)

        predictions = []
        previous_error = None
        for t in range(seq_len):
            h_0, c_0 = h_x
            cell_output = self.lstm(x_t=x[:, t], h_0=h_0, c_0=c_0, previous_error=previous_error)

            h_x = (cell_output['h'], cell_output['c'])

            head_input = torch.cat([cell_output[key] for key in self._head_inputs], dim=1)
            predictions.append(self.head(self._dropout(head_input)))

            if self._error_input:
                if y is not None:
                    previous_error = predictions[-1] - y[:, t]
                else:
                    LOGGER.warning('Ignoring input y since cfg.lstm_fewshot_error_input is False.')

        # stack to [batch size, sequence length, output_size]
        return torch.stack(predictions, 1)

    def copy_weights(self, base_lstm: Union[LSTM, ManualLSTM], copy_head: bool = True, copy_input_layer: bool = True):
        """Copy weights from a normal `LSTM` into this model class.

        Parameters
        ----------
        base_lstm : Union[LSTM, ManualLSTM]
            (Trained) instance of `tsfewshot.models.pytorchmodels.LSTM` or `tsfewshot.models.pytorchmodels.ManualLSTM`.
        copy_head : bool, optional
            If True, will also copy the model output layer's weights into this model.
        copy_input_layer : bool, optional
            If True, will also copy the model input layer's weights into this model.
        """
        assert isinstance(base_lstm, (LSTM, ManualLSTM))

        self.lstm.copy_weights(base_lstm.lstm, layer=0)

        if self.h_0 is not None:
            # copy learned initial states
            assert base_lstm.h_0 is not None
            self.h_0.data = copy.deepcopy(base_lstm.h_0).data
            self.c_0.data = copy.deepcopy(base_lstm.c_0).data

        if copy_head:
            self.head.load_state_dict(copy.deepcopy(base_lstm.head.state_dict()))
        if copy_input_layer:
            self.input_layer.load_state_dict(copy.deepcopy(base_lstm.input_layer.state_dict()))

    def reset_parameters(self):
        self.input_layer.reset_parameters()
        self.lstm.reset_parameters()
        self.head.reset_parameters()
        if self._learn_initial_state:
            self.h_0.data = torch.zeros((1, self.lstm.hidden_size), device=self.h_0.device)
            self.c_0.data = torch.zeros((1, self.lstm.hidden_size), device=self.c_0.device)


class _LSTMCell(nn.Module):
    """LSTM cell implementation that does not use nn.LSTM.

    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Hidden size of the model
    error_size : int, optional
        If cfg.lstm_fewshot_error_input is True, this value indicates how many additional inputs the cell will have.
        These additional inputs are meant to ingest the error on the previous timestep.
    initial_forget_bias : float, optional
        Initial bias of the forget gate.
    n_i_gates : int, optional
        Number of input gates, default 1.
    n_f_gates : int, optional
        Number of forget gates, default 1.
    n_g_gates : int, optional
        Number of inputs, default 1.
    n_o_gates : int, optional
        Number of output gates, default 1.
    """

    def __init__(self, input_size: int, hidden_size: int, error_size: int = 0, initial_forget_bias: float = 0,
                 n_i_gates: int = 1, n_f_gates: int = 1, n_g_gates: int = 1, n_o_gates: int = 1):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self._initial_forget_bias = initial_forget_bias

        self.w_hh = nn.Parameter(torch.zeros(4 * hidden_size, hidden_size), requires_grad=True)
        self.w_ih = nn.Parameter(torch.zeros(4 * hidden_size, input_size), requires_grad=True)
        self.b_hh = nn.Parameter(torch.zeros(4 * hidden_size), requires_grad=True)
        self.b_ih = nn.Parameter(torch.zeros(4 * hidden_size), requires_grad=True)

        # additional weights for error input
        self.w_eh = nn.Parameter(torch.zeros(4 * hidden_size, error_size), requires_grad=True)

        # additional weights and biases for additional forget and output gates.
        # These are separate parameters so we can individually enable/disable their training.
        self._n_i_gates = n_i_gates
        self._n_f_gates = n_f_gates
        self._n_g_gates = n_g_gates
        self._n_o_gates = n_o_gates
        n_additional = n_i_gates + n_f_gates + n_g_gates + n_o_gates - 4
        self.additional_w_hh = nn.Parameter(torch.zeros(n_additional * hidden_size, hidden_size), requires_grad=True)
        self.additional_w_ih = nn.Parameter(torch.zeros(n_additional * hidden_size, input_size), requires_grad=True)
        self.additional_w_eh = nn.Parameter(torch.zeros(n_additional * hidden_size, error_size), requires_grad=True)
        self.additional_b_hh = nn.Parameter(torch.zeros(n_additional * hidden_size), requires_grad=True)
        self.additional_b_ih = nn.Parameter(torch.zeros(n_additional * hidden_size), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        """Initializate model weights. """
        stdv = math.sqrt(3 / self.hidden_size)
        for weight in self.parameters():
            if len(weight.shape) > 1:
                weight.data.uniform_(-stdv, stdv)
            else:
                nn.init.zeros_(weight)

        if self._initial_forget_bias != 0:
            self.b_hh.data[self.hidden_size:2 * self.hidden_size] = self._initial_forget_bias
            if self._n_f_gates > 1:
                self.additional_b_hh.data[(self._n_i_gates - 1) * self.hidden_size:
                                          (self._n_i_gates + self._n_f_gates - 2) * self.hidden_size] = \
                    self._initial_forget_bias

    def forward(self, x_t: torch.Tensor, h_0: torch.Tensor, c_0: torch.Tensor,
                previous_error: torch.Tensor = None) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:

        # combine normal weights with weights for additional gates
        w_hh, w_ih, w_eh, b_hh, b_ih = [torch.cat([t, additional_t], dim=0) for t, additional_t in
                                        zip([self.w_hh, self.w_ih, self.w_eh, self.b_hh, self.b_ih],
                                            [self.additional_w_hh, self.additional_w_ih, self.additional_w_eh,
                                             self.additional_b_hh, self.additional_b_ih])]

        if previous_error is not None and not torch.isnan(previous_error).any():
            x_t = torch.cat([x_t, previous_error], dim=1)
            w_ih = torch.cat([w_ih, w_eh], dim=1)
        gates = h_0 @ w_hh.T + b_hh + x_t @ w_ih.T + b_ih

        gates = gates.chunk(self._n_i_gates + self._n_f_gates + self._n_g_gates + self._n_o_gates, 1)
        i = [gates[0]]
        f = [gates[1]]
        g = [gates[2]]
        o = [gates[3]]

        # additional gates are at the end because we concatenated their weights to the normal weights.
        i += gates[4:4 + self._n_i_gates - 1]
        f += gates[4 + self._n_i_gates - 1:
                   4 + self._n_i_gates + self._n_f_gates - 2]
        g += gates[4 + self._n_i_gates + self._n_f_gates - 2:
                   4 + self._n_i_gates + self._n_f_gates + self._n_g_gates - 3]
        o += gates[4 + self._n_i_gates + self._n_f_gates + self._n_g_gates - 3:]

        i_out = 1
        for i_k in i:
            i_out = i_out * torch.sigmoid(i_k)

        f_out = c_0
        for f_k in f:
            f_out = f_out * torch.sigmoid(f_k)

        g_out = 1
        for g_k in g:
            g_out = g_out * torch.tanh(g_k)

        c_1 = f_out + i_out * g_out
        h_1 = torch.tanh(c_1)
        for o_k in o:
            h_1 = h_1 * torch.sigmoid(o_k)

        return {'h': h_1, 'c': c_1, 'i': i, 'f': f[0], 'g': g, 'o': o[0],
                'f_additional': f[1:], 'o_additional': o[1:], 'i_additional': i[1:], 'g_additional': g[1:]}

    def copy_weights(self, base_lstm: Union[nn.Module, _LSTMCell], layer: int):
        """Copy weights from an `nn.LSTM` layer into this class.

        Parameters
        ----------
        base_lstm : Union[nn.Module, _LSTMCell]
            (Trained) instance of nn.LSTM.
        layer : int
            Which layer from base_lstm to copy.
        """

        assert self.hidden_size == base_lstm.hidden_size
        assert self.input_size == base_lstm.input_size

        if isinstance(base_lstm, _LSTMCell):
            self.w_hh.data = copy.deepcopy(base_lstm.w_hh).data
            self.w_ih.data = copy.deepcopy(base_lstm.w_ih).data
            self.b_hh.data = copy.deepcopy(base_lstm.b_hh).data
            self.b_ih.data = copy.deepcopy(base_lstm.b_ih).data
            self.w_eh.data = copy.deepcopy(base_lstm.w_eh).data
            if base_lstm.additional_w_hh.shape == self.additional_w_hh.shape:
                self.additional_w_hh.data = copy.deepcopy(base_lstm.additional_w_hh).data
                self.additional_w_ih.data = copy.deepcopy(base_lstm.additional_w_ih).data
                self.additional_w_eh.data = copy.deepcopy(base_lstm.additional_w_eh).data
                self.additional_b_hh.data = copy.deepcopy(base_lstm.additional_b_hh).data
                self.additional_b_ih.data = copy.deepcopy(base_lstm.additional_b_ih).data
            elif base_lstm.additional_w_hh.shape[0] == 0:
                pass  # base_lstm had no additional gates
            else:
                raise NotImplementedError('Copying weights with different numbers of gates is not implemented.')
        else:
            self.w_hh.data = copy.deepcopy(getattr(base_lstm, f'weight_hh_l{layer}').data)
            self.w_ih.data = copy.deepcopy(getattr(base_lstm, f'weight_ih_l{layer}').data)
            self.b_hh.data = copy.deepcopy(getattr(base_lstm, f'bias_hh_l{layer}').data)
            self.b_ih.data = copy.deepcopy(getattr(base_lstm, f'bias_ih_l{layer}').data)
