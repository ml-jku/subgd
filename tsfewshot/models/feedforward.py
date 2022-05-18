import torch

from tsfewshot.config import Config
from tsfewshot.models.basemodel import BasePytorchModel
from tsfewshot.models.inputoutputnet import InputOutputNet


class FeedForward(BasePytorchModel):
    """Fully-connected feed-forward network.

    This model consists only of an input and an output layer, with no "core" model in between.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)

        self.input_layer = InputOutputNet(cfg, usage='input')
        self.head = InputOutputNet(cfg, usage='output', input_size=self.input_layer.output_size)

    def forward(self, x: torch.Tensor, support_x: torch.Tensor = None, support_y: torch.Tensor = None) -> torch.Tensor:
        return self.head(self.input_layer(x))

    def reset_parameters(self):
        """Reset the parameters of the network. """
        super().reset_parameters()
        self.input_layer.reset_parameters()
        self.head.reset_parameters()
