import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from joblib import Parallel, delayed
from matplotlib.collections import LineCollection
from torch.nn.utils import parameters_to_vector
from tqdm import tqdm

from tsfewshot.config import Config
from tsfewshot.models import get_model
from tsfewshot.pcautils import get_best_epoch, load_diffs
from tsfewshot.utils import load_model, update_classification_head

LOGGER = logging.getLogger(__name__)


def parameter_style(name: str, typ='lstm') -> Tuple[str, str, str]:
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    colors = ['#003f5c', '#ffa600', 'C2', 'C3', 'C4', 'C5']
    if typ == 'lstm':
        labels = {'lstm.weight_ih_l0': ('LSTM input-hidden', colors[0], '-'),
                  'lstm.weight_hh_l0': ('LSTM hidden-hidden', colors[1], '-'),
                  'lstm.bias_ih_l0': ('LSTM bias', colors[3], '--'),
                  'head.networks.0.fc.0.weight': ('head weight', colors[2], '-'),
                  'head.networks.0.fc.0.bias': ('head bias', colors[4], '--')
                  }
    else:
        labels = {'head.networks.0.fc.0.weight': ('layer 1 weight', colors[0], '-'),
                  'head.networks.0.fc.0.bias': ('layer 1 bias', colors[1], '--'),
                  'head.networks.0.fc.3.weight': ('layer 2 weight', colors[2], '-'),
                  'head.networks.0.fc.3.bias': ('layer 2 bias', colors[3], '--'),
                  'head.networks.0.fc.6.weight': ('layer 3 weight', colors[4], '-'),
                  'head.networks.0.fc.6.bias': ('layer 3 bias', colors[5], '--'),
                  }
    return labels[name]

def plot_support_vs_mse_rlc(df: pd.DataFrame, support_sizes: List[int],
                            exclude_types: List[str] = None,
                            title=None,
                            aggregation='median',
                            metric_name='RMSE',
                            style=None,
                            alpha=0.6,
                            ax=None,
                            figsize=(15 * 1 / 2.54, 8 * 1 / 2.54), blogpost=False):
    if not blogpost:
        type_axes_list = [['no-finetune supervised', 'jfr', 'normal supervised', 'pca supervised'],
                     ['no-finetune reptile', 'normal reptile', 'pca reptile'], 
                     ['no-finetune fomaml', 'normal fomaml', 'pca fomaml']] # 'normal metasgd', 'normal metacurv'
        axes_titles = ['Supervised pre-training', 'Reptile', 'foMAML']
        if ax is None:
            f, axs = plt.subplots(1, 3, figsize=figsize, sharey=True)
            f.suptitle(title)
        else:
            f = None
            axs = [ax]
            plt.setp(axs[0], xticks=support_sizes)
    else:
        type_axes_list = [['no-finetune supervised', 'normal supervised', 'pca supervised']]
        axes_titles = ['Supervised pre-training']
        figsize=(7,4)
        f, axs = plt.subplots(1, 1, figsize=figsize, sharey=True)
        f.suptitle(title)
        axs = [axs]
        # plt.setp(axs[0], xticks=support_sizes)


    rank_vals = df.groupby('support', axis=1).rank(axis=1)
    vals = df.groupby(['support', 'type'], axis=1)
    if aggregation == 'median':
        vals = vals.agg(lambda s: s.median(skipna=False))
    elif aggregation == 'mean':
        vals = vals.agg(lambda s: s.mean(skipna=False))
    else:
        raise ValueError('Unknown aggregation')
    


    for i, axes_types in enumerate(type_axes_list):
        for typ in axes_types: #df.columns.get_level_values('type').unique():
            typ_aggregated = vals.loc[:, (slice(None), typ)]
            if aggregation == 'median':
                typ_aggregated = typ_aggregated.median(axis=0)
                typ_vals = df.stack(level='seed').loc[:, (slice(None), typ)]
                upper = typ_vals.quantile(0.75, axis=0)
                lower = typ_vals.quantile(0.25, axis=0)
            elif aggregation == 'mean':
                typ_aggregated = typ_aggregated.mean(axis=0)
                typ_vals = df.stack(level='seed').loc[:, (slice(None), typ)]
                ci = 1.96 * typ_vals.std(axis=0) / np.sqrt(typ_vals.shape[0])
                lower, upper = typ_aggregated - ci, typ_aggregated + ci

            ls, col, label, marker = get_style(typ)
            if style is not None and typ in style.keys():
                ls = style[typ].get('ls', ls)
                col = style[typ].get('col', col)
                label = style[typ].get('label', label)
                marker = style[typ].get('marker', marker)

            if exclude_types is not None and (label in exclude_types or typ in exclude_types):
                continue

            axs[i].plot(typ_aggregated.index.get_level_values('support'),
                        typ_aggregated, ls=ls, label=label, c=col, marker=marker)
            if not blogpost:
                axs[i].fill_between(typ_aggregated.index.get_level_values('support'),
                                    lower, upper, color=col, alpha=alpha)
            axs[0].set_ylabel(f'{"Average" if aggregation == "mean" else "Median"} {metric_name}')
            axs[i].set_title(axes_titles[i])
            

    for ax in axs:
        plt.setp(ax, xticks=support_sizes)
        ax.grid(alpha=0.6)
        ax.set_xlabel('Support size')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
    plt.figlegend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 0.93), ncol=5)
    plt.yscale('log')
    _ = plt.tight_layout()
    return f


def plot_support_vs_mse(df: pd.DataFrame, support_sizes: List[int], ranks: bool = False,
                        exclude_types: List[str] = None,
                        title=None,
                        aggregation='median',
                        metric_name='RMSE',
                        style=None,
                        alpha=0.6,
                        ax=None,
                        figsize=(15 * 1 / 2.54, 8 * 1 / 2.54)):
    if ax is None:
        f, axs = plt.subplots(1, 1, figsize=figsize)
        f.suptitle(title)
        axs = [axs]
    else:
        f = None
        axs = [ax]
    plt.setp(axs[0], xticks=support_sizes)

    rank_vals = df.groupby('support', axis=1).rank(axis=1)
    vals = df.groupby(['support', 'type'], axis=1)
    if aggregation == 'median':
        vals = vals.agg(lambda s: s.median(skipna=False))
    elif aggregation == 'mean':
        vals = vals.agg(lambda s: s.mean(skipna=False))
    else:
        raise ValueError('Unknown aggregation')

    for typ in df.columns.get_level_values('type').unique():
        typ_aggregated = vals.loc[:, (slice(None), typ)]
        if aggregation == 'median':
            typ_aggregated = typ_aggregated.median(axis=0)
            typ_vals = df.stack(level='seed').loc[:, (slice(None), typ)]
            upper = typ_vals.quantile(0.75, axis=0)
            lower = typ_vals.quantile(0.25, axis=0)
        elif aggregation == 'mean':
            typ_aggregated = typ_aggregated.mean(axis=0)
            typ_vals = df.stack(level='seed').loc[:, (slice(None), typ)]
            ci = 1.96 * typ_vals.std(axis=0) / np.sqrt(typ_vals.shape[0])
            lower, upper = typ_aggregated - ci, typ_aggregated + ci

        ls, col, label, marker = get_style(typ)
        if style is not None and typ in style.keys():
            ls = style[typ].get('ls', ls)
            col = style[typ].get('col', col)
            label = style[typ].get('label', label)
            marker = style[typ].get('marker', marker)

        if exclude_types is not None and (label in exclude_types or typ in exclude_types):
            continue

        if ranks:
            typ_ranks = rank_vals.loc[:, (slice(None), typ)]
            ranks_agg = typ_ranks.mean(axis=0) if aggregation == 'mean' else typ_ranks.median(axis=0)
            axs[0].plot(typ_ranks.columns.get_level_values('support'),
                        ranks_agg,
                        ls=ls, label=label, c=col, marker=marker)
            axs[0].fill_between(typ_ranks.columns.get_level_values('support'),
                                ranks_agg - typ_ranks.std(axis=0),
                                ranks_agg + typ_ranks.std(axis=0),
                                color=col, alpha=alpha)
            axs[0].set_ylabel(f'{aggregation} rank')
        else:
            axs[0].plot(typ_aggregated.index.get_level_values('support'),
                        typ_aggregated, ls=ls, label=label, c=col, marker=marker)
            axs[0].fill_between(typ_aggregated.index.get_level_values('support'),
                                lower, upper, color=col, alpha=alpha)
            axs[0].set_ylabel(f'{"Average" if aggregation == "mean" else "Median"} {metric_name}')

    for ax in axs:
        ax.grid(alpha=0.6)
        ax.set_xlabel('Support size')
        ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=3)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    _ = plt.tight_layout()
    return f


def get_style(typ):
    ls = '-'
    col = None
    marker = ''
    if 'no-finetune' in typ:
        label = 'no finetuning'
        col = '#003f5c'
        marker = ''
        ls = '--'
    elif 'pca' in typ:
        label = 'PCA'
        col = '#ffa600'
        marker = 'x'
        if 'mean' in typ:
            label += ' mean'
            col = 'C5'
        if 'noweight' not in typ:
            label += ' (weighted)'
            marker = '+'
            col = '#003f5c'
    elif 'adam' in typ and not 'pca' in typ and 'interpolate' not in typ:
        label = 'Adam'
        marker = 'v'
        col = '#ef5675'
    elif 'normal' in typ and 'pca' not in typ:
        label = 'SGD'
        marker = 'x'
        ls = '--'
        col = '#7a5195'
        if 'metasgd' in typ:
            label = 'Meta-SGD'
            col = 'red'
    elif 'random' in typ:
        label = 'Random'
        marker = 'd'
        ls = '-.'
        col = 'gray'
    elif 'fomaml' in typ.lower():
        label = 'foMAML'
        col = 'red'
        ls = '-'
        marker = 'x'
    elif 'metasgd' in typ.lower():
        label = 'MetaSGD'
        col = 'green'
        ls = '-'
        marker = 'v'
    elif 'jfr' in typ:
        label = 'JFR'
        col = '#2aa22a'
        ls = '--'
        marker = 'v'
    else:
        raise ValueError(f'Unknown type {typ}')

    if 'threshold' in typ:
        if 'random' not in typ:
            ls = '--'
        label += typ.split(' sparsity: ')[1]
    return ls, col, label, marker


def plot_deltas_rank(cfg: Config, finetune_dirs: List[Path], tols: List[float],
                     random_baseline: bool = True,
                     n_repeats: int = 1,
                     reduce_head: int = None,
                     ax=None,
                     task_steps: int = 5,
                     within_task_steps: int = None,
                     gradient_steps: int = None,
                     epoch_steps: int = None,
                     colors: List[str] = None,
                     plot_val_metric: bool = False,
                     use_erank: bool = False,
                     label_suffix: str = '',
                     init_epoch: int = -1,
                     plot_details: bool = True,
                     **kwargs):
    """Plot the rank of the stacked weight differences between pretraining and finetuning.

    This method plots the "rank" (actually, the number of singular values above a certain threshold) for the matrix of
    stacked weight differences from different tasks. Either it uses the global difference from pretraining-
    to finetuning-optimum, or the differences along the finetuning path at a given granularity.

    The plot shows how the rank increases with an increasing amount of tasks (and, therefore, increasing number of
    vectors).

    Parameters
    ----------
    cfg : Config
        The configuration of the pretraining run.
    finetune_dirs : List[Path]
        Paths to the finetuning runs (usually one for each task) from which the deltas will be calculated.
    tols : List[float]
        List of rank tolerances.
    random_baseline : bool, optional
        If True, will also plot the rank of a random matrix of the same shape for comparison.
    n_repeats : int, optional
        Number of repeated rank calculations. In each repetition, the order in which tasks are added to the matrix is
        randomly permuted and a new random matrix is drawn if `random_baseline` is True.
    reduce_head : int, optional
        If provided, will reduce the pretraining head output dimension to the specified dimension. Useful for N-way
        scenarios that were pretrained on ``M != N`` classes.
    ax : optional
        If provided, will plot on this axis, else, will create a new figure.
    task_steps : int, optional
        If this value is larger than 1, the rank will only be evaluated at points where ``i * task_steps`` tasks are
        included in the matrix.
    within_task_steps : int, optional
        If provided, will also calculate the ranks when subsequently adding blocks of `within_task_steps` vectors
        to the matrix that all belong to the same task.
    gradient_steps : int, optional
        If provided, the weight deltas will be calculated along the full gradient path (which must have been stored
        during finetuning via ``cfg.store_training_path``), where every `gradient_steps`-th step is used to calculate
        deltas.
    epoch_steps : int, optional
        If provided, the weight deltas will be calculated at every `epoch_steps`-th epoch.
    colors : List[str], optional
        List of colors for each tolerance.
    use_erank : bool, optional
        If True, will calculate effective rank instead of normal tolerance-based rank.
    init_epoch : int, optional
        Epoch to use as baseline for difference calculations.
    plot_val_metric : bool, optional
        If True, will plot the validation metric curve alongside the rank plot. Note that metrics are only available at
        the end of each epoch (or less, depending on ``cfg.eval_every``).
    plot_details : bool, optional
        If True, will print experiment details in small text in the plot.
    kwargs : dict
        Additional arguments will be passed to the plot function.

    Returns
    -------
    Matplotlib figure and axis.
    """
    cfg = Config(cfg.run_dir / 'config.yml')
    cfg.update({'layer_per_dataset_eval': False, 'device': 'cpu'})
    if reduce_head is not None:
        if cfg.classification_n_classes['train'] == reduce_head:
            raise ValueError('Head already has the right size')
        cfg.update({'classification_n_classes': {'train': cfg.classification_n_classes['train'],
                                                 'finetune': reduce_head}})
    uninit_model = get_model(cfg, is_test=True, is_finetune=False)
    init_model, _ = load_model(cfg, uninit_model, epoch=init_epoch)
    init_vector = parameters_to_vector(init_model.parameters()).detach().cpu()

    cfg.update({'layer_per_dataset': None, 'device': 'cpu'})
    LOGGER.info(f'{len(finetune_dirs)} finetune dirs.')
    diffs = Parallel(n_jobs=1, verbose=1)(delayed(load_diffs)(cfg, f, init_vector,
                                                              improved_epochs=None,
                                                              use_path=epoch_steps,
                                                              use_steps=gradient_steps,
                                                              reduce_classification_head=reduce_head)
                                          for f in finetune_dirs)
    diffs = [torch.stack(d) for d in diffs]  # type: ignore
    diffs = [d / torch.linalg.norm(d, dim=1, ord=2, keepdim=True) for d in diffs]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1 + int(plot_val_metric), 1, 1)
    else:
        fig = plt.gcf()
    LOGGER.info(f'diffs shapes {[tuple(d.shape) for d in diffs]}')

    def _erank(deltas):
        """Calculates the effective rank of a matrix. """
        assert deltas.ndim == 2
        singular_vals = np.linalg.svd(deltas, compute_uv=False)

        # normalizes input s -> scale independent!
        return torch.distributions.Categorical(torch.from_numpy(singular_vals)).entropy().exp().numpy()  # type: ignore

    def _matrix_ranks(matrix: Union[torch.Tensor, np.ndarray], tols: List[float], erank: bool) -> List[int]:
        """Calculate matrix rank for different tolerances with one SVD calculation. """
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.numpy()
        if erank:
            if len(tols) != 1:
                raise ValueError('erank has no tolerance')
            return [_erank(matrix)]
        singular_vals = np.linalg.svd(matrix, compute_uv=False)
        return [(singular_vals > tol).sum() for tol in tols]

    x_ticks = []
    first_permutation = None
    for i in range(n_repeats):
        np.random.seed(i)
        permutation = np.random.permutation(len(diffs))
        if first_permutation is None:
            first_permutation = permutation  # remember first permutation for plotting of val metrics
        matrix_sizes = []
        ranks = {tol: [] for tol in tols}
        random_ranks = {tol: [] for tol in tols}
        for k in tqdm(list(range(0, len(permutation), task_steps))):
            selected_diffs = torch.cat([diffs[p] for p in permutation[:k + 1]])
            if random_baseline:
                random_matrix = torch.normal(mean=0, std=1, size=selected_diffs.shape)
            if i == 0:
                x_ticks.append(selected_diffs.shape[0])

            last_task_size = diffs[permutation[k]].shape[0]
            if within_task_steps is not None and within_task_steps < last_task_size:
                inner_steps = tqdm(list(range(within_task_steps, last_task_size + 1, within_task_steps)))
            else:
                inner_steps = [last_task_size + 1]  # all steps
            for j, inner_step in enumerate(inner_steps):
                step_diffs = selected_diffs[:selected_diffs.shape[0] - last_task_size + inner_step]
                matrix_sizes.append(step_diffs.shape[0])
                j_ranks = _matrix_ranks(step_diffs, tols=tols, erank=use_erank)
                for tol, j_rank in zip(tols, j_ranks):
                    ranks[tol].append(j_rank)

                if random_baseline:
                    j_random_matrix = random_matrix[:step_diffs.shape[0]]  # type: ignore
                    j_random_matrix = j_random_matrix / torch.linalg.norm(j_random_matrix, dim=1, ord=2, keepdim=True)
                    j_random_ranks = _matrix_ranks(j_random_matrix, tols=tols, erank=use_erank)
                    for tol, j_rank in zip(tols, j_random_ranks):
                        random_ranks[tol].append(j_rank)

        for j, tol in enumerate(tols):
            col = colors[j] if colors is not None else f'C{j}'
            tol_label = f' (tol {tol})' if not use_erank else ''
            ax.plot(matrix_sizes, ranks[tol], c=col,
                    label=f'{tol_label}{label_suffix}' if i == 0 else None, **kwargs)
            if random_baseline:
                ax.plot(matrix_sizes, random_ranks[tol], c=col, alpha=0.8, ls='--',
                        label=f'Gaussian random{tol_label}' if i == 0 else None)

    ax.set_xlabel('Number of vectors')
    ax.set_ylabel('Rank' if not use_erank else 'Effective rank')
    ax.grid(True, alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1), ncol=2, mode='expand')
    if plot_details:
        ax.text(0, 1, f'{cfg.run_dir.name}\n'
                f'{finetune_dirs[0].parent.name} ({len(finetune_dirs)} runs)\n'
                f'task_steps {task_steps} within_task_steps {within_task_steps} '
                f'gradient_steps {gradient_steps} epoch_steps {epoch_steps}',
                fontsize=3, clip_on=True, transform=ax.transAxes, verticalalignment='top')

    if plot_val_metric:
        # TODO not sure if all cases are caught
        if gradient_steps is not None or within_task_steps != 1 or epoch_steps != 1 or task_steps != 1:
            raise NotImplementedError()
        fig.set_size_inches(10, 8)
        val_ax = fig.add_subplot(2, 1, 2, sharex=ax)
        val_metrics = []
        for i in first_permutation:  # type: ignore
            finetune_dir = finetune_dirs[i]
            best_epoch = get_best_epoch(finetune_dir)
            for epoch in range(1, best_epoch + 1):
                val_metrics.append(pd.read_csv(finetune_dir /
                                               f'results/test-model_epoch{str(epoch).zfill(3)}-{cfg.metric[0]}.csv',
                                               index_col=0).mean()[cfg.metric[0]])
        val_ax.plot(range(1, len(val_metrics) + 1), val_metrics)
        val_ax.set_xlabel(ax.get_xlabel())
        ax.set_xlabel('')
        val_ax.set_ylabel(cfg.metric[0])
        val_ax.grid(True)

    return fig, ax


def plot_gridsearch(gridsearch_results, ylim=None, aggregation='median'):
    n_plots = len([0 for x in gridsearch_results.values() if x is not None])
    f, ax = plt.subplots(n_plots, 1, figsize=(8, 4 * n_plots), sharex=True, sharey=True)
    if n_plots == 1:
        ax = [ax]
    i = 0
    for type_name, type_results in gridsearch_results.items():
        if type_results is None:
            continue
        type_results = type_results.copy()
        type_results.columns = pd.MultiIndex.from_tuples(type_results.columns, names=['type', 'dataset', 'run', 'seed'])
        type_results = type_results.groupby('type', axis=1)
        if aggregation == 'median':
            type_results = type_results.median()
        elif aggregation == 'mean':
            type_results = type_results.mean()
        else:
            raise ValueError('Unknown aggregation')
        for config in type_results.columns:
            ax[i].plot(type_results[config], label=config)
        ax[i].legend(frameon=False, loc=(1, 0))
        ax[i].set_title(type_name)
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(f'{aggregation} metric')
        if ylim is not None:
            ax[i].set_ylim(ylim)
        i += 1

    return f


def plot_path(cfg: Config, runs: Dict[str, Path], pca_file: Path, reduce_head: int = None):
    n_components = 8
    f, ax = plt.subplots(n_components, 2 * len(runs), figsize=(10 * len(runs), 12), sharex=True, sharey=True)
    cmaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

    pca = pickle.load(pca_file.open('rb'))
    pca_transform = pca['v'].detach().cpu()

    cfg = Config(cfg.run_dir / 'config.yml')
    cfg.update({'layer_per_dataset_eval': False, 'device': 'cpu'})
    if reduce_head is not None:
        cfg.update({'classification_n_classes': {'train': cfg.classification_n_classes['train'],
                                                 'finetune': reduce_head}})
    uninit_model = get_model(cfg, is_test=True, is_finetune=False)
    init_model, _ = load_model(cfg, uninit_model, epoch=-1)
    init_vector = parameters_to_vector(init_model.parameters()).detach().cpu()

    weight_indices = None
    cfg.update({'layer_per_dataset': None, 'device': 'cpu'})
    for k, (run_type, type_run) in enumerate(runs.items()):
        # for i, ds_run in enumerate(type_run):
        for i, ds_run in enumerate(sorted(list((type_run / 'finetuned_models').glob('*')))[:10]):
            weight_path = [init_vector]
            transformed_path = [pca_transform.T.mv(init_vector)]
            epoch = -np.inf
            for model_path in sorted(list(ds_run.glob('model_epoch*.p'))):
                new_epoch = int(model_path.name.split('epoch')[1][:-2])
                assert new_epoch > epoch
                epoch = new_epoch
                uninit_model = get_model(cfg, is_test=True, is_finetune=False)
                # average the head into a single set of weights, then replicate for the number of classes
                if reduce_head is not None:
                    uninit_model = update_classification_head(cfg, uninit_model)
                path_model, _ = load_model(cfg, uninit_model, model_path=model_path)
                path_weights = parameters_to_vector(path_model.parameters()).detach().cpu()
                weight_path.append(path_weights)
                transformed_path.append(pca_transform.T.mv(path_weights))

            transformed_path = torch.stack(transformed_path).numpy()
            transformed_path -= transformed_path[0]
            weight_path = torch.stack(weight_path).numpy()
            if weight_indices is None:
                weight_indices = np.argsort(weight_path.std(axis=0))[::-1]
            weight_path = weight_path[:, weight_indices]
            weight_path -= weight_path[0]
            for offset, path, name in zip([0, len(runs)],
                                          [transformed_path, weight_path],
                                          ['PCA-space', 'weight-space']):
                for j in range(n_components):
                    _colorline(path[:, 0], path[:, j + 1],
                               ax=ax[j, k + offset], alpha=1, lw=1.5, cmap=cmaps[i % len(cmaps)])
                    ax[j, k + offset].scatter(path[0:1, 0], path[0:1, j + 1], zorder=9999999,
                                              marker='x', s=15, c='black', label='start' if i == 0 else None)

                    ax[0, k + offset].set_title(f'{run_type}, {name}')
                    ax[-1, k + offset].set_xlabel('component 1' if offset == 0 else 'weight 1')
                    ax[j, offset].set_ylabel(f'component {j+2}' if offset == 0 else f'weight {j+2}')
                    ax[j, k + offset].legend(ncol=1, frameon=True)
                    ax[j, k + offset].grid(alpha=0.6)
                    if j > 0:
                        ax[j, k + offset].get_shared_x_axes().join(ax[j, k + offset], ax[j - 1, k + offset])
                    if k > 0:
                        ax[j, k + offset].get_shared_x_axes().join(ax[j, k + offset], ax[j, k - 1 + offset])
                        ax[j, k + offset].get_shared_y_axes().join(ax[j, k + offset], ax[j, k - 1 + offset])

    return f


def _make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


def _colorline(x, y, z=None, ax=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), lw=3, alpha=1.0):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''

    # colors
    if z is None:
        z = np.log(np.linspace(1, 1000, len(x))) / np.log(1000)

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = _make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=lw, alpha=alpha)

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)

    return lc
