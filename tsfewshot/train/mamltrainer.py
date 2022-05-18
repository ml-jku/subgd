import logging

import higher
import torch
from torch import optim

from tsfewshot.config import Config
from tsfewshot.train.episodictrainer import EpisodicTrainer

LOGGER = logging.getLogger(__name__)


class MAMLTrainer(EpisodicTrainer):
    """Class to train a PyTorch model in a MAML setup.

    This class will train a model in the MAML setup as in Algorithm 2 of [#]_.
    If cfg.maml_first_order is True, the higher-order gradients will be ignored (First-Order MAML).

    Parameters
    ----------
    cfg : Config
        Run configuration
    is_finetune : bool, optional
        Finetune an existing model on a full dataset. The difference to eval with finetune: true is that
        here we don't finetune in a few-shot setting but in normal supervised training with a train and validation set.

    References
    ----------
    .. [#] C. Finn, P. Abbeel, and S. Levine, "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks,"
           in Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia,
           6-11 August 2017, 2017, vol. 70, pp. 1126-1135. http://proceedings.mlr.press/v70/finn17a.html.
    """

    def __init__(self, cfg: Config, is_finetune: bool = False):
        super().__init__(cfg, is_finetune=is_finetune)

        self._first_order = cfg.maml_first_order
        self._n_inner_iter = cfg.maml_n_inner_iter

        if self._clip_gradient_norm is not None:
            LOGGER.warning('MAML uses gradient clipping by values, not by norm')

        self._batchnorm_states = [k for k in self._model.state_dict().keys()
                                  if 'running_mean' in k or 'running_var' in k]
        LOGGER.info(f'Found {len(self._batchnorm_states)} batchnorm states in the model.')

    def _train_epoch(self, epoch: int):

        episode_iter = iter(self._loaders)

        inner_optimizer = self._get_inner_optimizer()
        self._optimizer.zero_grad()
        query_losses = []
        for _ in range(self._batch_size):
            # sample two random sets query examples and support examples
            support_batch, query_batch = next(episode_iter)
            if isinstance(support_batch['x'], torch.Tensor) and isinstance(query_batch['x'], torch.Tensor):
                support_x = support_batch['x'].to(self._device)
                query_x = query_batch['x'].to(self._device)
            else:
                support_x = support_batch['x']
                query_x = query_batch['x']
            support_y = support_batch['y'].to(self._device)
            query_y = query_batch['y'].to(self._device)

            query_loss_kwargs = {}
            support_loss_kwargs = {}
            if 'std' in query_batch.keys():
                query_loss_kwargs['std'] = query_batch['std'].to(self._device)
                support_loss_kwargs['std'] = support_batch['std'].to(self._device)

            if self.noise_sampler_y is not None:
                support_y = support_y + self.noise_sampler_y.sample(support_y.shape).to(support_y)
                query_y = query_y + self.noise_sampler_y.sample(query_y.shape).to(query_y)

            # can't use cuDNN for MAML (unless it's first-order) because it doesn't support double backward pass
            with torch.backends.cudnn.flags(enabled=self._first_order):  # type: ignore
                # implementation adapted from
                # https://github.com/facebookresearch/higher/blob/master/examples/maml-omniglot.py
                with higher.innerloop_ctx(self._model, inner_optimizer, copy_initial_weights=False,
                                          track_higher_grads=not self._first_order) as (fnet, diffopt):

                    self._innerloop_hook(fnet, diffopt)

                    # Optimize the likelihood of the support set by taking gradient steps w.r.t. the model's
                    # parameters. This adapts the model's meta-parameters to the task. higher is able to
                    # automatically keep copies of your network's parameters as they are being updated.
                    for _ in range(self._n_inner_iter):
                        support_pred = fnet(support_x)  # type: ignore
                        support_loss = self._loss(support_pred, support_y, **support_loss_kwargs)
                        diffopt.step(support_loss)

                        self._tb_logger.log_step(maml_inner_loss=support_loss.item())

                    if self._cfg.batch_norm_mode in ['maml-conventional', 'metabn']:
                        # set batchnorm mode to eval
                        fnet.train(mode=False, update_batch_norm_only=True)

                    # The final set of adapted parameters will induce some final loss on the query dataset.
                    # These will be used to update the model's meta-parameters.
                    query_pred = fnet(query_x)  # type: ignore
                    query_loss = self._loss(query_pred, query_y, **query_loss_kwargs) / self._batch_size
                    query_losses.append(query_loss.detach())

                    # Update the model's meta-parameters to optimize the query losses across all of the tasks
                    # sampled in this batch. This unrolls through the gradient steps.
                    if self._first_order:
                        grads = torch.autograd.grad(query_loss, fnet.parameters())  # type: ignore
                        for grad, param in zip(grads, self._model.parameters()):
                            if param.grad is None:
                                param.grad = grad
                            else:
                                param.grad.data += grad
                    else:
                        query_loss.backward()

                    self._update_batchnorm(fnet)

                    self._tb_logger.log_step(loss=query_loss.item() * self._batch_size)

        if self._clip_gradient_norm is not None:
            torch.nn.utils.clip_grad_value_(self._model.parameters(), self._clip_gradient_norm)  # type: ignore

        self._optimizer.step()
        query_losses = sum(query_losses)

        if not self._cfg.silent:
            LOGGER.info(f'Epoch {epoch} query loss: {query_losses:.4f}')

    def _update_batchnorm(self, fnet):
        """Copy running means and variances back to the original model. """
        if len(self._batchnorm_states) > 0:
            fnet_state = fnet.state_dict()
            # load with strict = False because we only want to load the batchnorm states, not all states.
            self._model.load_state_dict({k: fnet_state[k] for k in self._batchnorm_states}, strict=False)

    def _get_inner_optimizer(self):
        if not isinstance(self._cfg.maml_inner_lr, float):
            raise ValueError('MAML inner lr must be a float')
        return optim.SGD(self._model.parameters(), self._cfg.maml_inner_lr)

    def _innerloop_hook(self, fnet, diffopt):
        """Hook for subclasses, e.g., MetaSGD. """
