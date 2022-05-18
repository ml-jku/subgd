import torch
from sklearn.linear_model import LinearRegression

from tsfewshot.config import Config
from tsfewshot.models.basemodel import BaseSklearnModel


class Persistence(BaseSklearnModel):
    """Baseline model that simply uses the current time step as the next prediction. """

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        if cfg.encoding_type == 'sum':
            raise ValueError('Cannot perform persistence prediction when positional encoding is added to inputs.')

        if cfg.input_vars['train'] != cfg.input_vars['finetune']:
            raise ValueError('Persistence will not work if inputs change during finetuning/evaluation.')
        if list(cfg.target_vars['train']) != list(cfg.target_vars['finetune']):
            raise ValueError('Persistence will not work if targets change during finetuning/evaluation.')

        # Store the indices into the input that will contain the target (there could be positional encoding and other
        # inputs that we need to ignore). We can't predict if inputs are deltas and output are values or vice versa.
        self._output_idxs = []
        for target in cfg.target_vars['train'].keys():
            if target not in cfg.input_vars['train']:
                raise ValueError('Persistence cannot predict a target that is not part of the inputs')

            input_position = cfg.input_vars['train'].index(target)
            if cfg.input_output_types['output'] in ['deltas', 'delta_t']:
                if cfg.input_output_types['input'] == 'values':
                    raise ValueError('Persistence cannot predict deltas from values.')
                if cfg.input_output_types['input'] == 'both':
                    self._output_idxs.append(len(cfg.input_vars['train']) + input_position)  # skip the absolute values
                else:
                    self._output_idxs.append(input_position)
            else:
                if cfg.input_output_types['input'] == 'deltas':
                    raise ValueError('Persistence cannot predict values from deltas.')
                self._output_idxs.append(input_position)

    def forward(self, x: torch.Tensor, support_x: torch.Tensor = None, support_y: torch.Tensor = None) -> torch.Tensor:
        # if positional encoding is used, we only want to return the actual inputs, not the encoding.
        # if inputs/targets are values/deltas/both, we need to select the right input indices.
        return x[:, :, self._output_idxs]

    def fit(self, x: torch.Tensor, y: torch.Tensor):
        """Does nothing. Required for API compatibility. """


class SupportMean(BaseSklearnModel):
    """Baseline model that simply uses the mean of the support set as the next prediction. """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self._predict_last_n = cfg.predict_last_n

    def forward(self, x: torch.Tensor, support_x: torch.Tensor = None, support_y: torch.Tensor = None) -> torch.Tensor:
        if support_y is None:
            raise ValueError('Need support set for predictions.')

        if support_y.shape[2] > 1:
            raise NotImplementedError('Mean-persistence is only implemented for single-target configs.')

        return torch.full((x.shape[0], x.shape[1], 1),
                          torch.mean(support_y[:, -self._predict_last_n:, :])).to(x.device)  # type: ignore

    def fit(self, x: torch.Tensor, y: torch.Tensor):
        """Does nothing. Required for API compatibility. """


class Linear(BaseSklearnModel):
    """Linear regression model. """

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self._output_size = cfg.output_size

        self.model = LinearRegression(n_jobs=cfg.num_workers)  # type: ignore

    def forward(self, x: torch.Tensor, support_x: torch.Tensor = None, support_y: torch.Tensor = None) -> torch.Tensor:
        x_in = x.reshape(-1, x.shape[2]).cpu().numpy()
        prediction = torch.from_numpy(self.model.predict(x_in)).to(x)

        return prediction.reshape((x.shape[0], x.shape[1], self._output_size))

    def fit(self, x: torch.Tensor, y: torch.Tensor):
        """Fit the linear regression model.

        Parameters
        ----------
        x : torch.Tensor
            Input data of shape (batch_size, seq_length, features)
        y : torch.Tensor
            Target data of shape (batch_size, seq_length, output_size)
        """
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        self.model.fit(x.cpu().numpy(), y.cpu().numpy())
