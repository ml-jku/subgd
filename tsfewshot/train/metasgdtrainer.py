import logging

from torch import optim

from tsfewshot.train.mamltrainer import MAMLTrainer

LOGGER = logging.getLogger(__name__)


class MetaSGDTrainer(MAMLTrainer):
    """Class to train a PyTorch model in a MetaSGD setup.

    This class will train a model in the MetaSGD setup as in [#]_.

    Parameters
    ----------
    cfg : Config
        Run configuration
    is_finetune : bool, optional
        Finetune an existing model on a full dataset. The difference to eval with finetune: true is that
        here we don't finetune in a few-shot setting but in normal supervised training with a train and validation set.

    References
    ----------
    .. [#] Li, Zhenguo, et al. "Meta-sgd: Learning to learn quickly for few-shot learning."
           arXiv preprint arXiv:1707.09835 (2017).
    """

    def _train_epoch(self, epoch: int):
        for lr in self._model.learning_rates:
            lr.requires_grad = True
        super()._train_epoch(epoch)
        for lr in self._model.learning_rates:
            lr.requires_grad = False  # turn off for evaluation

    def _get_inner_optimizer(self):
        # lr is part of the model, so this lr can be set to 1
        return optim.SGD(self._model.model.parameters(), 1.0)

    def _innerloop_hook(self, fnet, diffopt):
        # callback to apply per-parameter learning rate of MetaSGD
        def grad_callback(all_grads):
            return [g * lr for g, lr in zip(all_grads, self._model.learning_rates)]
        diffopt._grad_callback = grad_callback
