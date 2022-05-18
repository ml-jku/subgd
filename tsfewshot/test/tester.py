import copy
import logging
import re
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from matplotlib import figure
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from tsfewshot import utils
from tsfewshot.config import Config
from tsfewshot.data import get_dataset
from tsfewshot.data.basedataset import BaseDataset
from tsfewshot.data.episodicdataset import EpisodicTestDataset
from tsfewshot.logger import Logger
from tsfewshot.metrics import get_metrics
from tsfewshot.models import get_model
from tsfewshot.models.basemodel import BaseModel

LOGGER = logging.getLogger(__name__)


class Tester:
    """Class to test a model.

    For each target task, the tester samples cfg.support_size support time series and lets the model generate
    predictions for the remaining samples (queries).

    Parameters
    ----------
    cfg : Config
        Run configuration
    split : {'train', 'val', 'test'}
        Period to evaluate on.
    init_model_params : bool, optional
        If True, will try to load a model from disk if `model` is None. If False, will use the uninitialized model
        that is returned by ``get_model``.
    tb_logger : Logger, optional
        Initialized tensorboard logger.
    """

    def __init__(self, cfg: Config, split: str, init_model_params: bool = True, tb_logger: Logger = None):
        utils.set_seed(cfg.seed)

        self._init_model_params = init_model_params
        scaler = None
        if not init_model_params:
            # This can be reasonable, e.g., for a linear regression model that will be fit to the support set
            cfg = utils.setup_run_dir(cfg)
            utils.setup_logging(str(cfg.run_dir / 'output.log'))
            cfg.dump_config(cfg.run_dir / 'config.yml')
            LOGGER.warning('Warning: evaluation without prior training.')

            if len(cfg.train_datasets) > 0:
                LOGGER.info('Calculating scaler on train datasets')
                # this will dump the scaler to disk so we can access it later.
                _ = get_dataset(cfg, 'train', is_train=True, silent=cfg.silent)
            else:
                LOGGER.warning('No training data specified. Will train model without normalization.')
                scaler = BaseDataset.DUMMY_SCALER

            if tb_logger is None:
                tb_logger = Logger(cfg)
                tb_logger.start_tb()

        if scaler is None:
            scaler = BaseDataset.load_scaler_from_disk(cfg)

        self._cfg = cfg
        self._tb_logger = tb_logger

        self.noise_sampler_y = None
        if cfg.target_noise_std > 0:
            self.noise_sampler_y = torch.distributions.Normal(loc=0, scale=cfg.target_noise_std)

        self._n_way = cfg.classification_n_classes['finetune']
        if self._n_way is None:
            self._n_way = 1  # regression is like 1-way classification
        self._split = split
        if split == 'train':
            datasets = cfg.train_datasets
        elif split == 'val':
            datasets = cfg.val_datasets
        elif split == 'test':
            datasets = cfg.test_datasets
        else:
            raise ValueError(f'Unknown split {split}')

        self._datasets = {}
        LOGGER.info(f'Loading datasets (split={split})')
        for ds in (pbar := tqdm(datasets, file=sys.stdout, disable=cfg.silent)):
            pbar.set_postfix_str(ds)
            dataset = get_dataset(cfg, split=split, dataset=ds, is_train=False, train_scaler=scaler, silent=True)
            if len(dataset) == 0:
                LOGGER.warning(f'Dataset {ds} is empty. Skipping.')
                continue
            self._datasets[ds] = dataset

        self._target_vars = [f'{var}_{target}' for var, offsets in cfg.target_vars['finetune'].items()
                             for target in offsets]
        self._metrics = get_metrics(cfg, target_variables=self._target_vars,
                                    stds={name: ds.stds[name] for name, ds in self._datasets.items()
                                          if cfg.meta_dataset in ['camels', 'hbvedu']})

        self._device = cfg.device

    def subset_test_datasets(self, skip_indices: Dict[str, Sequence[int]]):
        """Restrict the tester to a subset of each test dataset.

        This method is used to create normal (non-metalearning) train-validation splits.

        Parameters
        ----------
        indices : Dict[str, Sequence[int]]
            For each dataset, a list of indices to skip in evaluation because they were used in training.
        """
        for ds_name, indices in skip_indices.items():
            if ds_name not in self._datasets:
                raise ValueError(f'Cannot subset {ds_name} because it is not an evaluation dataset.')

            dataset = self._datasets[ds_name]
            if len(dataset) - len(indices) < self._cfg.support_size:
                raise ValueError(f'{len(indices)} validation indices are not enough to create a subset.')
            if len(dataset) < max(indices):
                raise ValueError(f'Dataset {ds_name} is smaller than largest subset index.')
            remaining_indices = np.setdiff1d(range(len(dataset)), indices)
            LOGGER.info(f'Restricting evaluation of {ds_name} from {len(dataset)} to {len(remaining_indices)} samples.')
            self._datasets[ds_name] = Subset(dataset, remaining_indices)

    def evaluate(self, epoch: int = None, trained_model: BaseModel = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Evaluate the model without any tuning to the target dataset.

        This tester passes a support set to the model, which it can decide to use or not to use.

        Parameters
        ----------
        epoch : int, optional
            Epoch to evaluate. If -1 and `model` is None, will try to use the best epoch as stored in
            run_dir/best_epoch.txt. Default: last epoch saved to disk.
        trained_model : BaseModel, optional
            If provided, will evaluate the passed model (and not a model that's saved to disk). Overrides `epoch`.

        Returns
        -------
        pd.DataFrame
            DataFrame with one index entry per dataset and one column per target,
            plus one column with the metric across all targets.
        Optional[pd.Series]
            Series of metrics calculated globally across all evaluated datasets. None if metric cannot be calculated
            globally (e.g., NSE).
        """
        old_random_state = utils.get_random_states()

        if self._tb_logger is not None:
            self._tb_logger.eval()

        trained_model, model_name = self._setup_model(epoch, trained_model)

        eval_tasks = self._get_eval_tasks()
        eval_datasets = [{ds_name: self._datasets[ds_name] for ds_name in task} for task, _ in eval_tasks]
        eval_seeds = [seed for _, seed in eval_tasks]

        seen_tasks = {}
        global_metrics = None
        for i in range(len(eval_datasets)):
            # If seed isn't given by cfg.val_n_random_datasets, we draw query and support samples with
            # cfg.seed + epoch so they are the same across runs that use the same seed. Don't use a fixed
            # query/support set in all epochs to get a more general picture during validation.
            # We store and reset the seed after evaluation to make sure the remaining processing doesn't
            # become deterministic.
            # This could be implemented more nicely, but changes would likely lead to different randomness and
            # therefore to different query/support sets.

            # if a task is evaluated multiple times, change the seed to get different support/query sets
            seed_offset = 0
            if eval_tasks[i] in seen_tasks.keys():
                seed_offset = seen_tasks[eval_tasks[i]] + 1
            seen_tasks[eval_tasks[i]] = seed_offset

            # make sure seed is not negative
            utils.set_seed(self._cfg.seed + abs(epoch if epoch is not None else 0) + seed_offset)  # type: ignore
            eval_datasets[i] = EpisodicTestDataset(self._cfg, eval_datasets[i], seed=eval_seeds[i])  # type: ignore

        metric_names = self._cfg.metric
        if self._cfg.classification_n_classes['finetune'] is None:
            metric_names += [f'{t}_{m}' for t, m in product(self._target_vars, self._cfg.metric)]
        metrics = pd.DataFrame(columns=metric_names)

        backup_model = None
        if (self._cfg.layer_per_dataset == 'output' and self._cfg.layer_per_dataset_eval) \
                or self._cfg.classification_n_classes['train'] != self._cfg.classification_n_classes['finetune']:
            backup_model = copy.deepcopy(trained_model)

        n_figures = 0
        global_predictions, global_queries = [], []
        for dataset in (pbar := tqdm(eval_datasets, file=sys.stdout, disable=self._cfg.silent)):
            ds_name = dataset.name
            if ds_name in metrics.index:
                ds_name += '_1'  # in episodic n-way settings, a task may be sampled multiple times
                uniqueness_idx = 2
                while ds_name in metrics.index:
                    ds_name = ds_name.rsplit('_', maxsplit=1)[0] + f'_{uniqueness_idx}'
                    uniqueness_idx += 1
            pbar.set_postfix_str(ds_name)
            pbar.set_description_str('Setup')

            # select the correct set of output weights for the current dataset
            if self._cfg.layer_per_dataset == 'output' and self._cfg.layer_per_dataset_eval:
                train_rotations = [ds.split('#rotate', maxsplit=1)[1] for ds in self._cfg.train_datasets]
                trained_model = utils.reduce_per_dataset_layer(
                    self._cfg, backup_model,
                    choose=train_rotations.index(ds_name.split('#rotate', maxsplit=1)[1]))

            support_x, support_y, support_sample_ids, support_offsets, support_y_base, query_loader = \
                self._get_query_support(epoch, dataset)

            if self._tb_logger is not None:
                self._tb_logger.finetune()
            pbar.set_description_str('Finetune')
            finetuned_model = self._finetune_hook(trained_model, ds_name, support_x=support_x, support_y=support_y,
                                                  support_supplemental={'y_base': support_y_base,
                                                                        'offset': support_offsets,
                                                                        'sample': support_sample_ids},
                                                  query_loader=query_loader)
            if self._tb_logger is not None:
                self._tb_logger.eval()
            finetuned_model.eval()

            # get rescaled absolute predictions/targets
            pbar.set_description_str('Predict')
            save_file = None
            if self._cfg.save_predictions:
                predictions_dir = self._cfg.run_dir / 'predictions'  # type: ignore
                predictions_dir.mkdir(parents=True, exist_ok=True)
                save_file = utils.get_file_path(predictions_dir,
                                                split=self._split,
                                                ds_name=ds_name,
                                                epoch=str(epoch),
                                                ext='nc')
            queries_y_abs, predictions_abs, queries_sample_ids, queries_offsets = \
                self._get_predictions(finetuned_model,
                                      ds_name=ds_name,
                                      query_loader=query_loader,
                                      support_x=support_x,
                                      support_y=support_y,
                                      save_file=save_file)
            global_predictions.append(predictions_abs)
            global_queries.append(queries_y_abs)
            pbar.set_description_str('Metrics')
            ds_metrics = self._get_metrics(ds_name, queries_y_abs, predictions_abs, queries_sample_ids)
            for key, metric in ds_metrics.items():
                metrics.loc[ds_name, key] = metric  # type: ignore

            plot = None
            if (self._cfg.plot_n_figures is None or n_figures < self._cfg.plot_n_figures) \
                    and self._cfg.classification_n_classes['finetune'] is None:
                pbar.set_description_str('Plot')
                n_figures += 1
                plot = self._plot_predictions(f'{ds_name} {self._cfg.metric[0]}: {ds_metrics[self._cfg.metric[0]]:.3f}',
                                              target=queries_y_abs,
                                              prediction=predictions_abs,
                                              query_sample_ids=queries_sample_ids,
                                              query_offsets=queries_offsets,
                                              support_sample_ids=support_sample_ids,
                                              support_offsets=support_offsets)
            if self._tb_logger is not None:
                self._tb_logger.log_step(dataset=ds_name, **metrics.loc[ds_name].to_dict())  # type: ignore
                if plot is not None:
                    self._tb_logger.log_figure(plot, f'{self._split}/timeseries')
            else:
                if plot is not None:
                    file_name = re.sub(r'[^A-Za-z0-9\._\-]+', '', f'{self._split}_{ds_name}_epoch{epoch}.png')
                    file_path = self._cfg.run_dir / 'plots' / file_name
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    plot.savefig(file_path, dpi=300, facecolor='w')
            plt.close()

        if len(eval_datasets) > 0:
            if self._tb_logger is not None:
                self._tb_logger.summarize()

            # save results to disk
            results_file = self._cfg.run_dir / 'results' / \
                f'{self._split}-{model_name}-{"_".join(self._cfg.metric)}.csv'
            results_file.parent.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(results_file)

            # calculate global metrics across all predictions.
            # can't do it for NSE, since it needs per-dataset std. Not a great solution, we could still calculate the
            # other metrics here
            if 'nse' not in self._cfg.metric:
                global_predictions = torch.cat(global_predictions, dim=0)
                global_queries = torch.cat(global_queries, dim=0)
                global_metrics = pd.Series(self._get_metrics('global_metrics',  # type: ignore
                                                             global_queries,
                                                             global_predictions,
                                                             timeseries_id=None))  # type: ignore
                global_results_file = self._cfg.run_dir / 'results' / \
                    f'global-{self._split}-{model_name}-{"_".join(self._cfg.metric)}.csv'
                global_metrics.to_csv(global_results_file)

        # restore random state
        utils.set_seed(old_random_state)

        return metrics, global_metrics

    def _get_eval_tasks(self) -> List[Tuple[Tuple[str], Optional[int]]]:
        # get possible N-way dataset combinations.
        # (Regression settings are 1-way settings, i.e., a task is a single dataset)
        # It doesn't matter which dataset we use to create the combinations; they are all of the same type
        if isinstance(self._cfg.val_n_random_datasets, list):
            # val_n_random_datasets is list of entries like [[ds1, ds2, ds3], seed].
            # the list of n datasets defines an n-way task, the seed defines the seed to use for drawing support
            # and query set for the task.
            # convert to tuple to make the list entries hashable. This also serves as a sanity check that the list
            # entries have roughly the right format.
            eval_tasks = [(tuple(task), seed) for task, seed in self._cfg.val_n_random_datasets]
        else:
            some_dataset = self._datasets[list(self._datasets.keys())[0]]
            if isinstance(some_dataset, Subset):
                # when train_val_split is set, the datasets in validation will be subsets
                some_dataset = some_dataset.dataset
            eval_tasks = [(task, None) for task in some_dataset.get_valid_dataset_combinations(  # type: ignore
                list(self._datasets.keys()), self._n_way, self._cfg.val_n_random_datasets)]

        return eval_tasks  # type: ignore

    def _get_query_support(self, epoch: Optional[int], ds: EpisodicTestDataset) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, DataLoader]:

        support_x, support_y, support_sample_ids, support_offsets, support_y_base = ds.get_support_set()

        # queries may be large, so we don't send to GPU yet
        # Can't use pin_memory for PyTorch < 1.8.0: https://github.com/pytorch/pytorch/pull/48543
        test_batchsize = 256
        query_loader = DataLoader(ds,
                                  batch_size=test_batchsize,
                                  num_workers=self._cfg.num_workers,
                                  persistent_workers=self._cfg.num_workers > 0,
                                  pin_memory=self._cfg.num_workers == 0)
        if self.noise_sampler_y is not None and support_y is not None:
            support_y = support_y + self.noise_sampler_y.sample(support_y.shape).to(support_y)
        if support_x is not None:
            if isinstance(support_x, torch.Tensor):
                support_x = support_x.to(self._device, non_blocking=True)
            support_y = support_y.to(self._device, non_blocking=True)
            support_y_base = support_y_base.to(self._device, non_blocking=True)
        return (support_x,
                support_y,
                support_sample_ids,
                support_offsets,
                support_y_base,
                query_loader)  # type: ignore

    def _get_predictions(self, finetuned_model: BaseModel,
                         ds_name: str,
                         query_loader: DataLoader,
                         support_x: torch.Tensor = None,
                         support_y: torch.Tensor = None,
                         save_file: Path = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            prediction = []
            queries_y = []
            query_y_base = []
            query_sample_ids = []
            query_offsets = []
            for query_batch in query_loader:
                queries_y.append(query_batch['y'])
                query_y_base.append(query_batch['y_base'])
                query_sample_ids.append(query_batch['sample'])
                query_offsets.append(query_batch['offset'])
                if self._cfg.lstm_finetune_error_input:
                    # also provide target values so the model can calculate the error on the previous timesteps
                    prediction.append(finetuned_model(query_batch['x'].to(self._device),
                                                      support_x=support_x,
                                                      support_y=support_y,
                                                      y=queries_y[-1].to(self._device)).cpu())
                elif self._cfg.model == 'eulerode':
                    prediction.append(finetuned_model(query_batch['x'],
                                                      support_x=support_x,
                                                      support_y=support_y).cpu())
                else:
                    prediction.append(finetuned_model(query_batch['x'].to(self._device),
                                                      support_x=support_x,
                                                      support_y=support_y).cpu())

            queries_y = torch.cat(queries_y, dim=0)
            query_y_base = torch.cat(query_y_base, dim=0)
            query_sample_ids = torch.cat(query_sample_ids, dim=0)
            query_offsets = torch.cat(query_offsets, dim=0)

            # doesn't matter which dataset we use for rescaling, because they all use the same scaler from training
            some_dataset = self._datasets[list(self._datasets.keys())[0]]
            if isinstance(some_dataset, Subset):
                # when train_val_split is set, the datasets in validation will be subsets
                some_dataset = some_dataset.dataset
            prediction = some_dataset.rescale_targets(torch.cat(prediction, dim=0))  # type: ignore
            queries_y = some_dataset.rescale_targets(queries_y)  # type: ignore

            if self._cfg.input_output_types['output'] in ['deltas', 'delta_t']:
                # convert deltas into absolute values.
                # y_base was never normalized, so it doesn't need to be rescaled.
                if query_y_base is None:
                    raise ValueError('Need y_base to convert to absolute values.')
                prediction = query_y_base + prediction
                queries_y = query_y_base + queries_y

            if save_file is not None:
                prediction_dict = {'sample_id': (['sample'], query_sample_ids),
                                   'offset': (['sample'], query_offsets)}
                if prediction.shape[2] == queries_y.shape[2]:
                    prediction_dict.update({k: (['sample', 'step', 'variable'], v)
                                            for k, v in zip(['prediction', 'target'], [prediction, queries_y])})
                else:
                    # in classification, predictions have a larger dim 2 than targets
                    prediction_dict.update({'prediction': (['sample', 'step', 'variable_prediction'], prediction)})
                    prediction_dict.update({'target': (['sample', 'step', 'variable'], queries_y)})
                prediction_xr = xr.Dataset(prediction_dict, attrs={'dataset': ds_name})
                prediction_xr.to_netcdf(save_file)

            return queries_y, prediction, query_sample_ids, query_offsets

    def _get_metrics(self, ds_name: str,
                     queries_y: torch.Tensor,
                     prediction: torch.Tensor,
                     timeseries_id: torch.Tensor) -> Dict[str, float]:
        metrics = {}
        key = ds_name
        if key not in self._datasets and key != 'global_metrics' and 'nse' in self._cfg.metric:
            # if val_n_random_datasets is set, the dataset can have a suffix "_\d+" which we have to remove
            # for the std lookup.
            key, suffix = ds_name.rsplit('_', maxsplit=1)
            if not all(s in '0123456789' for s in suffix):
                raise RuntimeError(f'Unknown dataset {ds_name}')  # this should never happen
        for metric_fn in self._metrics:
            metrics.update(metric_fn(prediction, queries_y, dataset=key, timeseries_id=timeseries_id))
        return metrics

    def _setup_model(self, epoch: int = None, trained_model: BaseModel = None) -> Tuple[BaseModel, str]:
        """Load model from file if not provided, create a string identifier for the model. """
        if trained_model is None:
            if self._init_model_params:
                # FinetuneTester will reload the model with is_finetune=True.
                # We load with is_finetune=False here to make sure we get the model in its training configuration,
                # and not a possibly altered finetuning setup into which we could not load the weights.
                uninit_model = get_model(self._cfg, is_test=True, is_finetune=False).to(self._cfg.device)
                trained_model, model_name = utils.load_model(self._cfg, uninit_model, epoch)
                trained_model = trained_model.to(self._device)
            else:
                if epoch is not None:
                    raise ValueError('evaluate was provided an epoch but it uses an uninitialized model.')
                LOGGER.warning('Warning: using model without loading its parameters')
                trained_model = get_model(self._cfg, is_test=True, is_finetune=False).to(self._device)
                model_name = 'raw'
        else:
            if epoch is None:
                LOGGER.warning('Passed a model but not epoch. May overwrite results if called repeatedly.')
                model_name = 'passed_model'
            else:
                model_name = f'model_epoch{str(epoch).zfill(3)}'

            if self._cfg.layer_per_dataset is not None and not self._cfg.layer_per_dataset_eval:
                # reduce the per-dataset layer from training into one averaged layer
                trained_model = utils.reduce_per_dataset_layer(self._cfg, trained_model)
            if self._cfg.classification_n_classes['train'] != self._cfg.classification_n_classes['finetune']:
                # update the head size to the new number of classes
                trained_model = utils.update_classification_head(self._cfg, trained_model)

        return trained_model, model_name

    def _plot_predictions(self,
                          title: str,
                          target: torch.Tensor,
                          prediction: torch.Tensor,
                          query_sample_ids: torch.Tensor,  # trajectory 'axis'
                          query_offsets: torch.Tensor,  # sample 'axis in data array
                          support_sample_ids: torch.Tensor = None,
                          support_offsets: torch.Tensor = None) -> figure.Figure:

        if self._cfg.model == 'eulerode':
            fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(8, 4))
            # plot only the first item of the query set, i.e. target[0]
            ax0.plot(target[0], alpha=0.6, c='b', label='target')
            ax0.plot(prediction[0], alpha=0.6, c='r', ls='--', label='prediction')

            ax0.grid(True)
            ax0.set_xlabel('step')
            ax0.set_ylabel('output')
            # ax0.legend()

            ax1.plot(target[0], alpha=0.6, c='b', label='target')
            ax1.plot(prediction[0], alpha=0.6, c='r', ls='--', label='prediction')
            ax1.set_xlim(600, 800)
            ax1.grid(True)
            ax1.set_xlabel('step')
            ax1.set_ylabel('output')
            # ax1.legend()

            ax2.plot(target[0], alpha=0.6, c='b', label='target')
            ax2.plot(prediction[0], alpha=0.6, c='r', ls='--', label='prediction')
            ax2.set_xlim(1000, 1200)
            ax2.grid(True)
            ax2.set_xlabel('step')
            ax2.set_ylabel('output')
            ax2.legend()

            return fig

        fig, axis = plt.subplots()
        if self._cfg.timeseries_is_sample or self._cfg.predict_last_n > 1:
            LOGGER.warning('Plotting only supports timeseries_is_sample = False / predict_last_n = 1.')
            return fig

        max_sample = int(query_sample_ids.max()) + 1
        rand_samples = np.random.choice(range(max_sample), size=min(5, max_sample), replace=False)

        scatter = False
        max_timestep = int(query_offsets.max())
        if support_offsets is not None:
            max_timestep = max(max_timestep, int(support_offsets.max()))
        # one extra step to separate samples
        max_timestep += 2

        ordered_target = np.full((len(rand_samples) * max_timestep, target.shape[2]), np.nan)
        ordered_pred = np.full_like(ordered_target, np.nan)
        for i, sample in enumerate(rand_samples):
            if len(rand_samples) > 1 and not scatter:
                axis.axvline(i * max_timestep, c='black', lw=0.5)  # sample borders

            sample_target = target[query_sample_ids == sample]
            sample_pred = prediction[query_sample_ids == sample]
            for j, step in enumerate(query_offsets[query_sample_ids == sample]):
                ordered_target[i * max_timestep + step] = sample_target[j, -1].cpu().numpy()
                ordered_pred[i * max_timestep + step] = sample_pred[j, -1].cpu().numpy()
            if support_sample_ids is not None and support_offsets is not None:
                for j, step in enumerate(support_offsets[support_sample_ids == sample]):
                    axis.axvline(i * max_timestep + step, c='gray',  # type: ignore
                                 label='support' if i == 0 and j == 0 else None, alpha=.4, lw=0.3)

        for i, target_var in enumerate(self._target_vars):
            if scatter:
                axis.scatter(range(len(ordered_target)), ordered_target[:, i],
                             label=f'obs {target_var}', lw=1, s=5)
                axis.scatter(range(len(ordered_pred)), ordered_pred[:, i],
                             label=f'sim {target_var}', alpha=.8, lw=1, s=5)
            else:
                axis.plot(ordered_target[:, i], label=f'obs {target_var}', lw=1)
                axis.plot(ordered_pred[:, i], label=f'sim {target_var}', alpha=.8, lw=1, ls='--')

        box = axis.get_position()
        axis.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
        axis.set_title(title, wrap=True, size='small')
        return fig

    def _finetune_hook(self, model: BaseModel,
                       ds_name: str,
                       support_x: torch.Tensor = None,
                       support_y: torch.Tensor = None,
                       support_supplemental: Dict[str, torch.Tensor] = None,
                       query_loader: DataLoader = None) -> BaseModel:
        """Hook to finetune the model on the support set. Does nothing in the base tester class.

        Parameters
        ----------
        model : BaseModel
            The trained model to be finetuned.
        ds_name : str
            Name of the dataset that is being finetuned on (used for logging).
        support_x : torch.Tensor, optional
            Input data of the support set. Must be passed for finetuning.
        support_y : torch.Tensor, optional
            Target data of the support set. Must be passed for finetuning.
        support_supplemental : Dict[str, torch.Tensor], optional
            Dictionary with additional support set information ("sample", "offset", "y_base")
        query_loader : DataLoader, optional
            DataLoader of the query set. Required for semi-supervised/transductive finetuning strategies.

        Returns
        -------
        BaseModel
            The finetuned model (if finetuning is active, else the model remains unchanged).
        """
        return model
