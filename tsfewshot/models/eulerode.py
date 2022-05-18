from typing import Dict, List

import torch
from torch import nn
from tsfewshot.config import Config

from tsfewshot.models.basemodel import BasePytorchModel
from tsfewshot.utils import get_activation


class EulerODE(BasePytorchModel):
    """
    Implementation of the neural system identification / neural-ode apporach of the paper [#]_.

    References
    ----------
    .. [#] Forgione, Marco, et al. "On the adaptation of recurrent neural networks for system identification."
           arXiv preprint arXiv:2201.08660 (2022).
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.state_space_vars = cfg.eulerode_config['state_space']
        # get output indices, i.e. the state space variables corresponding to targets
        self.output_vars = list(cfg.target_vars['train'].keys())
        out_ss_idxes = []
        for out_var in self.output_vars:
            assert out_var in self.state_space_vars, f"Target variable {out_var} not part of the state variables."
            out_ss_idxes.append(self.state_space_vars.index(out_var))
        self.output_ss_idxes = out_ss_idxes
        # get state space dimension
        ss_dim = len(self.state_space_vars)
        # get state space model hidden feature dimension
        ss_hidden_size = cfg.hidden_size
        # get input dimension
        input_size = len(cfg.input_vars['train'])

        activation = cfg.eulerode_config['activation']

        if not (cfg.input_output_types['input'] == 'values' and cfg.input_output_types['output'] == 'values'):
            raise ValueError('For EulerODE inputs and outputs must be absolute values!')

        if cfg.seq_length != cfg.predict_last_n:
            raise ValueError('`seq_length` must be equal to `predict_last_n`!')

        ss_model = NeuralStateSpaceModel(n_x=ss_dim, n_u=input_size, n_feat=ss_hidden_size, activation=activation)
        self.forward_euler_simulator = ForwardEulerSimulator(ss_model)

        self.reset_parameters()

    def forward(self, x: Dict[str, torch.Tensor], support_x: torch.Tensor = None, support_y: torch.Tensor = None) -> torch.Tensor:
        x0 = x['x0'].to(device=self.device)
        u = x['u'].to(device=self.device)

        pred = self.forward_euler_simulator(x0, u)
        pred = pred[:, :, self.output_ss_idxes]  # predict only the output variables
        return pred

    def reset_parameters(self):
        super().reset_parameters()
        self.forward_euler_simulator.reset_parameters()


# adapted copy from here: https://github.com/forgi86/RNN-adaptation/blob/96834a12269187cbdb19f00509d4c8edc03c9dbc/torchid/statespace/module/ss_simulator_ct.py#L8
class ForwardEulerSimulator(nn.Module):
    r"""Forward Euler integration of a continuous-time neural state space model.

    Parameters
    ----------
    ss_model : nn.Module
        The neural SS model to be fitted
    ts : np.float
        model sampling time

    Examples
    --------

        >>> ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64)
        >>> nn_solution = ForwardEulerSimulator(ss_model)

    References
    ----------
    .. [#] https://github.com/forgi86/RNN-adaptation/

     """

    def __init__(self, ss_model, ts=1.0):
        super(ForwardEulerSimulator, self).__init__()
        self.ss_model = ss_model
        self.ts = ts

    def forward(self, x0_batch: torch.Tensor, u_batch: torch.Tensor) -> torch.Tensor:
        r"""Multi-step simulation over (mini-)batches

        Parameters
        ----------
        x0_batch : torch.Tensor (batch_size, 1, n_x)
            Initial state for each subsequence in the minibatch
        u_batch : torch.Tensor (batch_size, seq_len, n_u)
            Input sequence for each subsequence in the minibatch

        Returns
        -------
        torch.Tensor (batch_size, seq_len, n_x)
            Simulated state for all subsequences in the minibatch

        Examples
        --------

        >>> y_sim = nn_solution(x0, u)

        References
        ----------
        .. [#] https://github.com/forgi86/RNN-adaptation/
        """

        x_sim_list: List[torch.Tensor] = []
        x_step = x0_batch

        for u_step in u_batch.split(1, dim=1):  # i in range(seq_len):
            dx = self.ss_model(x_step, u_step)
            x_step = x_step + self.ts * dx
            x_sim_list += [x_step]  # contains system states after the input

        x_sim = torch.cat(x_sim_list, dim=1)
        return x_sim

    def reset_parameters(self):
        self.ss_model.reset_parameters()

# adapted copy from: https://github.com/forgi86/RNN-adaptation/blob/96834a12269187cbdb19f00509d4c8edc03c9dbc/torchid/statespace/module/ssmodels_ct.py#L8


class NeuralStateSpaceModel(nn.Module):
    r"""A state-space continuous-time model.

    Parameters
    ----------
    n_x : int 
        Number of state variables
    n_u : int
        Number of input variables
    n_feat : Optional[int]
        Number of input features in the hidden layer. Default: 64
    init_small : Optional[boolean]
        If True, initialize to a Gaussian with mean 0 and std 10^-4. Default: True
    activation : str
        Activation function in the hidden layer. Either 'relu', 'softplus', 'tanh'. Default: 'relu'

    Examples
    --------
        >>> ss_model = NeuralStateSpaceModel(n_x=2, n_u=1, n_feat=64)

    References
    ----------
    .. [#] https://github.com/forgi86/RNN-adaptation/
    """

    def __init__(self, n_x: int, n_u: int, n_feat: int = 64, scale_dx: float = 1.0, init_small: bool = True, activation: str = 'tanh'):
        super(NeuralStateSpaceModel, self).__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.n_feat = n_feat
        self.scale_dx = scale_dx
        self.init_small = init_small

        activation = get_activation(activation)

        self.net = nn.Sequential(
            nn.Linear(n_x + n_u, n_feat),  # 2 states, 1 input
            activation,
            nn.Linear(n_feat, n_x)
        )

    def forward(self, in_x, in_u):
        in_xu = torch.cat((in_x, in_u), -1)  # concatenate x and u over the last dimension to create the [xu] input
        dx = self.net(in_xu)  # \dot x = f([xu])
        dx = dx * self.scale_dx
        return dx

    def reset_parameters(self):
        # Small initialization is better for multi-step methods
        if self.init_small:
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)
