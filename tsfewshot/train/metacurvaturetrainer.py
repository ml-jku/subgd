import logging

import torch
from torch import optim

from tsfewshot.train.mamltrainer import MAMLTrainer

LOGGER = logging.getLogger(__name__)


class MetaCurvatureTrainer(MAMLTrainer):
    """Class to train a PyTorch model in a Meta-Curvature setup.

    This class will train a model in the Meta-Curvature setup as in [#]_.

    Parameters
    ----------
    cfg : Config
        Run configuration
    is_finetune : bool, optional
        Finetune an existing model on a full dataset. The difference to eval with finetune: true is that
        here we don't finetune in a few-shot setting but in normal supervised training with a train and validation set.

    References
    ----------
    .. [#] Park, Eunbyung, and Junier B. Oliva. "Meta-curvature."
           Advances in Neural Information Processing Systems 32 (2019).
    """

    def _train_epoch(self, epoch: int):
        self._model.toggle_grad(True)
        super()._train_epoch(epoch)
        self._model.toggle_grad(False)  # turn off for evaluation

    def _get_inner_optimizer(self):
        if not isinstance(self._cfg.maml_inner_lr, float):
            raise ValueError('Meta-Curvature inner lr must be a float')
        return optim.SGD(self._model.model.parameters(), self._cfg.maml_inner_lr)

    def _innerloop_hook(self, fnet, diffopt):
        # callback to apply preconditioning of Meta-Curvature
        def grad_callback(all_grads):
            return [torch.mm(m_out, torch.mm(g, m_in)) if g.ndim == 2 else m_in * g
                    for g, m_in, m_out in zip(all_grads, self._model.m_in, self._model.m_out)]
        diffopt._grad_callback = grad_callback
